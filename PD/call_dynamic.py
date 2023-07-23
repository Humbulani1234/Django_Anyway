from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput

graph = GraphvizOutput()
graph.output_file = "file4.png"

with PyCallGraph(output=graph):
  print("Hello world")
