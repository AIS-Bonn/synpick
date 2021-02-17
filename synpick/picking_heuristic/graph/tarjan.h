// Tarjan's strongly connected component algorithm
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef TARTJAN_H
#define TARTJAN_H

#include "graph.h"

namespace graph
{

typedef std::vector<VertexID> ConnectedComponent;

std::vector<ConnectedComponent> tarjan(const Graph& graph);
bool containsCycle(const Graph& graph, const std::vector<bool>& edgeMask);

}

#endif
