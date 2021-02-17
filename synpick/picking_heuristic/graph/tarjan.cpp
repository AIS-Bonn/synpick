// Tarjan's strongly connected component algorithm
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include "tarjan.h"

#include <stack>

namespace graph
{

namespace
{

class TarjanAlgo
{
public:
	TarjanAlgo(const Graph& graph, const std::vector<bool>* edgeMask = 0)
	 : m_graph(graph)
	 , m_vertexIndex(graph.vertices().size(), UNSET)
	 , m_vertexLowlink(graph.vertices().size(), UNSET)
	 , m_vertexOnStack(graph.vertices().size(), false)
	 , m_edgeMask(edgeMask ? *edgeMask : std::vector<bool>(graph.edges().size(), true))
	{}

	std::vector<ConnectedComponent> run()
	{
		for(VertexID v = 0; v < m_graph.vertices().size(); ++v)
		{
			if(m_vertexIndex[v] == UNSET)
				strongconnect(v);
		}

		return std::move(m_components);
	}

	bool containsCycle()
	{
		for(VertexID v = 0; v < m_graph.vertices().size(); ++v)
		{
			if(m_vertexIndex[v] == UNSET)
			{
				if(cycleCheck(v))
					return true;
			}
		}

		return false;
	}
private:
	void strongconnect(VertexID v)
	{
		m_vertexIndex[v] = m_index;
		m_vertexLowlink[v] = m_index;

		m_index++;

		m_S.push(v);
		m_vertexOnStack[v] = true;

		// Consider successors of v
		for(auto& edgeID : m_graph.vertices()[v].successors)
		{
			if(!m_edgeMask[edgeID])
				continue;

			auto& edge = m_graph.edges()[edgeID];
			VertexID w = edge.to;

			if(m_vertexIndex[w] == UNSET)
			{
				strongconnect(w);
				m_vertexLowlink[v] = std::min(m_vertexLowlink[v], m_vertexLowlink[w]);
			}
			else if(m_vertexOnStack[w])
			{
				// Successor w is in stack S and hence in the current SCC
				// Note: The next line may look odd - but is correct.
				// It says w.index not w.lowlink; that is deliberate and from the original paper
				m_vertexLowlink[v] = std::min(m_vertexLowlink[v], m_vertexIndex[w]);
			}
		}

		// If v is a root node, pop the stack and generate an SCC
		if(m_vertexLowlink[v] == m_vertexIndex[v])
		{
			ConnectedComponent comp;
			comp.reserve(m_S.size());

			VertexID w;
			do
			{
				w = m_S.top();
				m_S.pop();
				m_vertexOnStack[w] = false;
				comp.push_back(w);
			}
			while(w != v);

			m_components.push_back(std::move(comp));
		}
	}

	// Just check for cycles (early stopping)
	bool cycleCheck(VertexID v)
	{
		m_vertexIndex[v] = m_index;
		m_vertexLowlink[v] = m_index;

		m_index++;

		m_S.push(v);
		m_vertexOnStack[v] = true;

		// Consider successors of v
		for(auto& edgeID : m_graph.vertices()[v].successors)
		{
			if(!m_edgeMask[edgeID])
				continue;

			auto& edge = m_graph.edges()[edgeID];
			VertexID w = edge.to;

			if(m_vertexIndex[w] == UNSET)
			{
				if(cycleCheck(w))
					return true;

				m_vertexLowlink[v] = std::min(m_vertexLowlink[v], m_vertexLowlink[w]);
			}
			else if(m_vertexOnStack[w])
			{
				// Successor w is in stack S and hence in the current SCC
				// Note: The next line may look odd - but is correct.
				// It says w.index not w.lowlink; that is deliberate and from the original paper
				m_vertexLowlink[v] = std::min(m_vertexLowlink[v], m_vertexIndex[w]);
			}
		}

		// If v is a root node, pop the stack and generate an SCC
		if(m_vertexLowlink[v] == m_vertexIndex[v])
		{
			std::size_t size = 0;

			VertexID w;
			do
			{
				w = m_S.top();
				m_S.pop();
				m_vertexOnStack[w] = false;
				size++;
			}
			while(w != v);

			if(size != 1)
				return true; // cycle detected
		}

		return false;
	}

	const Graph& m_graph;

	VertexID m_index = 0;
	std::stack<VertexID> m_S;

	std::vector<VertexID> m_vertexIndex;
	std::vector<VertexID> m_vertexLowlink;
	std::vector<bool> m_vertexOnStack;

	std::vector<ConnectedComponent> m_components;

	std::vector<bool> m_edgeMask;
};

}

std::vector<ConnectedComponent> tarjan(const Graph& graph)
{
	TarjanAlgo algo(graph);
	return algo.run();
}

bool containsCycle(const Graph& graph, const std::vector<bool>& edgeMask)
{
	TarjanAlgo algo(graph, &edgeMask);
	return algo.containsCycle();
}

}
