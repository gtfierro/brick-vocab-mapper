from functools import cached_property
from pathlib import Path
from tqdm import tqdm
from loguru import logger as log
import os
from fnvhash import fnv, fnv1a_32
import numpy as np
from collections import defaultdict
from usearch.index import Index
import openai
from typing import Optional, List, Dict, Iterable
from rdflib import Graph, BRICK
from rdflib.term import Node
from dataclasses import dataclass, field


@dataclass(unsafe_hash=True)
class BrickClassDefinition:
    class_: Node
    label: str
    definition: str = ""
    definitions: List[Dict[str, str]] = field(
        default_factory=list, compare=False, hash=False
    )


class Mapper:
    def __init__(
        self,
        definitions: List[Dict[str, dict]],
        external_index_file: str = "external.index",
    ):
        # load in Brick
        self.g = Graph()
        self.g.parse(
            "https://github.com/BrickSchema/Brick/releases/download/nightly/Brick.ttl",
            format="ttl",
        )
        self.definitions = definitions
        self.brick_lookup = {}
        self.external_index_file: Path = Path(external_index_file)
        self.brick_index_file: Path = Path("brick.index")

        # 1536 is the openai embedding vector size
        self.external_index = Index(ndim=1536, metric="cos")
        self.brick_index = Index(ndim=1536, metric="cos")

        self.external_lookup = {
            int(fnv1a_32(d["name"].encode("utf8"))): d["name"] for d in self.definitions
        }

        self.populate_external_embeddings()

    def get_embedding(self, text: str) -> np.ndarray:
        return np.array(
            openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][
                0
            ]["embedding"]
        )  # type: ignore

    def populate_external_embeddings(self):
        """
        Calculate embeddings for the definitions passed in. Expect ['name'] key
        """
        if self.external_index_file.exists():
            log.info(f"Restored {self.external_index_file}")
            self.external_index.load(self.external_index_file)
            return

        log.info(
            f"No local 'self.external_index_file' found. Computing embeddings for external concepts"
        )
        for defn in tqdm(self.definitions):
            name = defn.pop("name")
            values = filter(lambda x: x and len(x), defn.values())
            text = f"{name} {' '.join(values)}".strip()
            defn["name"] = name
            self.external_index.add(
                int(fnv1a_32(name.encode("utf8"))), self.get_embedding(text)
            )
        self.external_index.save(self.external_index_file)

    def populate_brick_embeddings(self, brick_classes: Iterable[BrickClassDefinition]):
        """
        Calculates embeddings for the provided set of brick classes. Loads from 'brick.index' if it exists
        in the current directory

        :param brick_classes: list of BrickClassDefinition for the classes to be embedded
        """
        if self.brick_index_file.exists():
            self.brick_index.load(self.brick_index_file)
            for d in brick_classes:
                self._hashbrick(d.class_)
            return
        log.info("No local 'brick.index' found. Computing embeddings for Brick classes")
        for defn in tqdm(brick_classes):
            embedding = self.get_embedding(
                f"{defn.class_} {defn.label} {defn.definition}".strip()
            )
            key = self._hashbrick(defn.class_)
            self.brick_index.add(key, embedding)
        self.brick_index.save(self.brick_index_file)

    def _hashbrick(self, n: Node) -> int:
        hv = int(fnv1a_32(str(n).encode("utf8")))
        self.brick_lookup[hv] = n
        return hv

    def get_brick_classes(
        self, root: Node, skip_trees: Optional[List[Node]] = [BRICK.Parameter]
    ) -> List[BrickClassDefinition]:
        """
        Returns a list of Brick classes rooted (and including) 'root', and skipping all classes
        which are subclasses of (and including) each node in 'skip_trees'

        :param root: Parent class from which to fetch all Brick class definitions, including the parent
        :param skip_trees: List of Brick classes to skip (including all their children), defaults to [BRICK.Parameter]
        :return: all brick classes rooted at 'root' and not in a subtree rooted by anything in 'skip_trees'
        """
        query = f"""SELECT ?cls ?label ?definition WHERE {{
        ?cls a owl:Class ;
            rdfs:subClassOf* {root.n3()} ;
            rdfs:label ?label .
        OPTIONAL {{ ?cls skos:definition ?definition }}
        """
        if skip_trees is not None:
            for tree_root in skip_trees:
                query += (
                    f"FILTER NOT EXISTS {{ ?cls rdfs:subClassOf* {tree_root.n3()} }}\n"
                )
        query += "}"
        res = self.g.query(query)
        bcs = set()
        for class_, label, definition in res:
            bcs.add(
                BrickClassDefinition(
                    class_, str(label), str(definition) if definition else None
                )
            )
        return list(bcs)

    def _populate_external_to_brick_mapping(
        self, brick_classes: Iterable[BrickClassDefinition], top_k=20, threshold=0.8
    ) -> Dict[Node, List[dict]]:
        mapping = {}

        for defn in tqdm(brick_classes):
            # embedding = self.get_embedding(f"{defn.class_} {defn.label} {defn.definition}".strip())
            embedding = self.brick_index.get(
                np.array([self._hashbrick(defn.class_)], dtype=np.uint64)
            )
            if not embedding or embedding[0] is None:
                continue
            recommendations = self.external_index.search(embedding[0], count=top_k)
            best_labels = []
            for idx, external_match in enumerate(recommendations.keys):
                if recommendations.distances[idx] < threshold:
                    best_labels.append(external_match)
            kept_recs = [self.external_lookup[idx] for idx in best_labels]
            mapping[defn.class_] = kept_recs
        self.external_to_brick_mapping = mapping
        return mapping

    def get_mapping(
        self, bcs: List[BrickClassDefinition], allow_collisions: bool = True
    ) -> Dict[str, Node]:
        """
        Returns a mapping of external concepts -> brick concepts. If 'allow_collisions'
        is True, the mapping will allow multiple external concepts to map to multiple Brick concepts
        and vice-versa. If 'allow_collisions' is False, then the mapping will choose the best "stable"
        1-1 matching of external concepts and Brick concepts.

        :param allow_collisions: allow collisions in the final mapping, defaults to True
        :return: Mapping of External concepts (str) to Brick classes (rdflib.URIRef)
        """
        self.populate_brick_embeddings(bcs)
        if allow_collisions:
            return self._get_best_external_to_brick_mapping(bcs)
        join = self.external_index.join(self.brick_index)
        return {
            str(self.external_lookup[external_id]): self.brick_lookup[brick_id]
            for external_id, brick_id in join.items()
            if brick_id in self.brick_lookup
        }

    def _get_best_external_to_brick_mapping(
        self, brick_classes: Iterable[BrickClassDefinition]
    ) -> Dict[str, Node]:
        inverse_mapping = defaultdict(list)
        self._populate_external_to_brick_mapping(brick_classes)
        # for each external concept, make a list of all the Brick classes that
        # map to the concept
        for class_, candidates in self.external_to_brick_mapping.items():
            if not len(candidates):
                continue
            inverse_mapping[candidates[0]].append(class_)
        # figure out which Brick class has the highest score for each external concept
        singular_inverse_mapping: Dict[str, Node] = {}
        for cw_class, brick_classes in inverse_mapping.items():
            # sort based on decreasing vector distance ('score')
            brick_classes = sorted(brick_classes, key=lambda x: x[1], reverse=True)
            singular_inverse_mapping[cw_class] = brick_classes[0]
        return singular_inverse_mapping
