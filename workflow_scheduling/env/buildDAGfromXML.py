# assume the runtime in *.xml is the execution time on a CPU with 16 cores
# references: Characterizing and profiling scientific workflows

# networkx version
import networkx as nx
# import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def buildGraph(type, filename):
    tot_processTime = 0
    dag = nx.DiGraph(type=type)
    tree = ET.parse(filename)
    root = tree.getroot()
    for child in root:
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}job':
            size = 0
            for p in child:
                size += int(p.attrib['size'])
            dag.add_node(int(child.attrib['id'][2:]), processTime=float(child.attrib['runtime']) * 16, size=size)
            tot_processTime += float(child.attrib['runtime']) * 16
            # dag.add_node(child.attrib['id'], processTime=float(child.attrib['runtime'])*16, size=size)
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
            kid = int(child.attrib['ref'][2:])
            for p in child:
                parent = int(p.attrib['ref'][2:])
                dag.add_edge(parent, kid)
    return dag, tot_processTime

