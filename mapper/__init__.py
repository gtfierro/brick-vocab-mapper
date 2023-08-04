from tqdm import tqdm
from collections import defaultdict
from itertools import islice
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import openai
from typing import Optional, List, Generator, Dict, Iterable
from rdflib import Graph, BRICK
from rdflib.term import Node
from dataclasses import dataclass

# TODO:
# - use https://unum-cloud.github.io/usearch/#disk-based-indexes for storing/caching embeddings

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

@dataclass
class BrickClassDefinition:
    class_: Node
    label: str
    definition: str = ''


class Mapper:
    def __init__(self, definitions: List[Dict[str, dict]]):
        # load in Brick
        self.g = Graph()
        self.g.parse("https://github.com/BrickSchema/Brick/releases/download/nightly/Brick.ttl", format="ttl")
        self.definitions = definitions
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.recreate_collection(
            collection_name='External',
            vectors_config={
                'Description': rest.VectorParams(
                    distance=rest.Distance.COSINE,
                    size=1536, # openai embedding size
                ),
            }
        )
        self.populate_external_embeddings()

    def get_embedding(self, text: str) -> List[float]:
        return openai.Embedding.create(
            input=text, model="text-embedding-ada-002"
        )["data"][0]["embedding"]


    def get_external_embeddings(self) -> Generator[rest.PointStruct, None, None]:
        """
        Calculate embeddings for the definitions passed in. Expect ['name'] key
        """

        for defn in tqdm(self.definitions):
            name = defn.pop('name')
            values = filter(lambda x: x and len(x), defn.values())
            text = f"{name} {' '.join(values)}".strip()
            defn['name'] = name
            yield rest.PointStruct(
                id=hash(name),
                vector={'Description': self.get_embedding(text)},
                payload=defn,
            )

    def populate_external_embeddings(self):
        for batch in batched(self.get_external_embeddings(), 1):
            self.qdrant.upsert(
                collection_name='External',
                points=list(batch),
            )

    def get_brick_classes(self, root: Node, skip_trees: Optional[List[Node]] = None) -> Generator[BrickClassDefinition, None, None]:
        query = f"""SELECT ?cls ?label ?definition WHERE {{
        ?cls a owl:Class ;
            rdfs:subClassOf* {root.n3()} ;
            rdfs:label ?label .
        OPTIONAL {{ ?cls skos:definition ?definition }}
        """
        if skip_trees is not None:
            for tree_root in skip_trees:
                query += f"FILTER NOT EXISTS {{ ?cls rdfs:subClassOf* {tree_root.n3()} }}\n"
        query += "}"
        res = self.g.query(query)
        for (class_, label, definition) in tqdm(res):
            yield BrickClassDefinition(class_, str(label), str(definition) if definition else None)

    def populate_external_to_brick_mapping(self, brick_classes: Iterable[BrickClassDefinition], top_k=20, threshold=.8) -> Dict[Node, List[dict]]:
        mapping = {}
        for defn in brick_classes:
             embedding = self.get_embedding(f"{defn.class_} {defn.label} {defn.definition}".strip())
             recommendations = self.qdrant.search(collection_name='External',
                                                  query_vector=('Description', embedding),
                                                  limit=top_k)
             kept_recs = [d.dict() for d in recommendations if d.score > threshold]
             mapping[defn.class_] = kept_recs
        self.external_to_brick_mapping = mapping
        return mapping

    def get_best_external_to_brick_mapping(self, brick_classes: Iterable[BrickClassDefinition]) -> Dict[str, Node]:
        inverse_mapping = defaultdict(list)
        # for each external concept, make a list of all the Brick classes that
        # map to the concept
        for class_, candidates in self.external_to_brick_mapping.items():
            if not len(candidates):
                continue
            inverse_mapping[candidates[0]['payload']['name']].append((class_, candidates[0]['score']))
        # figure out which Brick class has the highest score for each external concept
        singular_inverse_mapping: Dict[str, Node] = {}
        for cw_class, brick_classes in inverse_mapping.items():
            # sort based on decreasing vector distance ('score')
            brick_classes = sorted(brick_classes, key=lambda x: x[1], reverse=True)
            singular_inverse_mapping[cw_class] = brick_classes[0][0]
        return singular_inverse_mapping



if __name__ == '__main__':
    import json
    defns = json.load(open('../../clockworksanalytics-brick/pointtypes.json'))
    fixed_defns = []
    for d in defns:
        name = d['attributes'].get('PointTypeName', '')
        display = d['attributes'].get('DisplayName', '')
        description = d['attributes'].get('PointTypeDescription', '')
        fixed_defns.append({'name': name, 'display': display, 'description': description})

    m = Mapper(fixed_defns)
    bcs = m.get_brick_classes(BRICK.Point, [BRICK.Parameter])
    m.populate_external_to_brick_mapping(bcs)
    for k, v in m.get_best_external_to_brick_mapping(bcs).items():
        print(k, v)
