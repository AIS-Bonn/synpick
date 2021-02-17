// Code adapted from the ARC 2017 system
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#include <torch/extension.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <stack>

#include <pybind11/stl_bind.h>

#include "pole_of_inaccessibility.h"

#include "graph/graph.h"
#include "graph/tarjan.h"

constexpr double TOTE_WIDTH = 0.615;
constexpr double LOW_SEGMENTATION_THRESHOLD = 0.3;
constexpr std::size_t LOW_SEGMENTATION_MIN_OBJECTS = 4;

static std::vector<cv::Scalar> COLORS{
    cv::Scalar(121,218,74),
    cv::Scalar(107,71,200),
    cv::Scalar(205,205,81),
    cv::Scalar(203,77,198),
    cv::Scalar(115,192,110),
    cv::Scalar(82,47,110),
    cv::Scalar(127,212,188),
    cv::Scalar(213,76,61),
    cv::Scalar(112,161,186),
    cv::Scalar(199,136,63),
    cv::Scalar(159,140,206),
    cv::Scalar(86,108,59),
    cv::Scalar(200,82,133),
    cv::Scalar(59,57,65),
    cv::Scalar(207,178,160),
    cv::Scalar(124,61,53)
};

namespace arc_perception
{

struct Detection
{
    std::string object;
    int classID = -1;

    cv::Mat_<uint8_t> binMask;
    cv::Point2i centroid;
    cv::Point2i suctionPoint;
    cv::Point2i polygonCentroid;
    double confidence = 0.0;
    double visibleAreaFactor = 0.0;
    double occlusionFactor = 0.0;
    int aboveBelow = 0;
    std::vector<std::vector<cv::Point>> contours;

    std::vector<std::string> objectsAbove;
    std::vector<double> objectsAboveWeight;

    unsigned int totalObjectsAbove = 0;

    Eigen::Vector3f cloudCentroid = Eigen::Vector3f::Zero();

    Eigen::Quaterniond predictedOrientation = Eigen::Quaterniond::Identity();

    uint32_t pixelCount;

    bool operator<(const Detection& other) const
    { return centroid.y < other.centroid.y; }
};


std::vector<Detection> postprocessSegmentation(const cv::Mat_<uint8_t>& segmentation, const cv::Mat_<float>& confidence, const std::vector<std::string>& classes, const std::vector<Eigen::Vector3f>& objectSizes, std::vector<double>& objectWeights)
{
    std::vector<Detection> detections;

    #pragma omp parallel for
    for(std::size_t classIdx = 0; classIdx < classes.size(); ++classIdx)
    {
        auto name = classes[classIdx];

        if(name == "box" || name == "unknown" || name == "vacuum_pipe")
            continue;

        cv::Mat_<uint8_t> classMask = (segmentation == classIdx);

        cv::Mat_<int32_t> label;
        cv::Mat_<int32_t> stats;
        cv::Mat_<double> centroids;
        cv::connectedComponentsWithStats(classMask, label, stats, centroids);

        if(stats.rows <= 1)
        {
//             printf("No pixels for class '%s'\n", classes[classIdx].c_str());
            continue;
        }

        // Find biggest component
        std::size_t biggestIdx = 1;
        for(int i = 2; i < stats.rows; ++i)
        {
            if(stats(i, cv::CC_STAT_AREA) > stats(biggestIdx, cv::CC_STAT_AREA))
                biggestIdx = i;
        }

        // Sum confidence
        cv::Mat_<uint8_t> objMask = (label == biggestIdx);
        cv::Mat_<float> confCrop = cv::Mat_<float>::zeros(objMask.rows, objMask.cols);
        confidence.copyTo(confCrop, objMask);

        double meanConf = cv::sum(confCrop)[0] / stats(biggestIdx, cv::CC_STAT_AREA);

        Detection det;
        det.object = name;
        det.classID = classIdx;
        det.binMask = objMask;
        det.confidence = meanConf;
        det.centroid.x = centroids(biggestIdx, 0);
        det.centroid.y = centroids(biggestIdx, 1);
        det.pixelCount = stats(biggestIdx, cv::CC_STAT_AREA);

        // Estimate expected area
        std::vector<float> dims{objectSizes[classIdx].x(), objectSizes[classIdx].y(), objectSizes[classIdx].z()};
        std::sort(dims.begin(), dims.end());

        float metricArea = dims[2] * dims[1];
        float pixelArea = metricArea * std::pow(std::max(objMask.cols, objMask.rows) / TOTE_WIDTH, 2.0);

        det.visibleAreaFactor = stats(biggestIdx, cv::CC_STAT_AREA) / pixelArea;

        std::vector<std::vector<cv::Point>> rawContours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(det.binMask, rawContours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point>> contours;

        // Add main contour
        contours.push_back(rawContours.at(0));
        if(cv::contourArea(contours[0], true) >= 0)
        {
            fprintf(stderr, "Exteriour contour seems to go the wrong way, I will disregard this detection.\n");
            continue;
        }

        // Add child contours
        bool ok = true;
        for(int child = hierarchy[0][2]; child >= 0; child = hierarchy[child][0])
        {
            auto contour = rawContours.at(child);
            if(cv::contourArea(contour, true) <= 0)
            {
                fprintf(stderr, "Exteriour contour seems to go the wrong way, I will disregard this detection.\n");
                ok = false;
                break;
            }

            contours.push_back(contour);
        }
        if(!ok)
            continue;

        det.contours = contours;

        cv::Point2d poleOfInacc = arc_perception::poleOfInaccessibility(contours);
        det.suctionPoint = poleOfInacc;

        // Also consider the centroid
        cv::Point2d centroid = arc_perception::polygonCentroid(contours.at(0));
        det.polygonCentroid = centroid;

        double poleDist = contourDistance(contours, poleOfInacc);

        double centroidFactor = arc_perception::contourDistance(contours, centroid) / poleDist;

        double distFactor = 0.8;

        // On heavy objects, we *really* prefer the centroid
        if(objectWeights[classIdx] > 0.8)
            distFactor = 0.4;

//         ROS_DEBUG("%s: centroid is %f of suction point, needs to be > %f", det.object.c_str(), centroidFactor, distFactor);

        if(centroidFactor > distFactor)
        {
//             ROS_DEBUG("Using centroid as suction point.");
            det.suctionPoint = centroid;
        }

        // Also consider the point between the centroid and the pole of inaccessibility
        cv::Point2d mid = (poleOfInacc + centroid)/2;
        double midFactor = arc_perception::contourDistance(contours, mid) / poleDist;

        if(centroidFactor > midFactor)
        {
            if(centroidFactor > distFactor)
            {
                printf("Using centroid as suction point.\n");
                det.suctionPoint = centroid;
            }
        }
        else
        {
            if(midFactor > distFactor)
            {
//                 printf("Using between point as suction point\n");
                det.suctionPoint = mid;
            }
        }

//         ROS_DEBUG(" %20s confidence %f", name.c_str(), meanConf);

        #pragma omp critical
        {
            detections.push_back(det);
        }
    }

    // Sort detections by confidence
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b){
        return a.confidence > b.confidence;
    });

    // How many would we throw away if we would apply the LOW_SEGMENTATION_THRESHOLD?
    std::size_t removal = 0;
    for(int i = detections.size()-1; i >= 0; --i)
    {
        if(detections[i].confidence >= LOW_SEGMENTATION_THRESHOLD)
            break;
        removal++;
    }

    std::size_t keep = detections.size() - removal;

    // Keep at least LOW_SEGMENTATION_MIN_OBJECTS
    keep = std::max(keep, std::min(LOW_SEGMENTATION_MIN_OBJECTS, detections.size()));

    detections.resize(keep);

    return detections;
}

static void trySolution(const graph::Graph& G, const std::vector<graph::EdgeID>& problematic, std::size_t i, std::vector<bool>* work, std::vector<bool>* bestSolution, double* bestWeight)
{
    // Are we still cyclic?
    if(!graph::containsCycle(G, *work))
    {
        // Sum up remaining edges
        double weight = 0.0;
        for(graph::EdgeID e = 0; e < G.edges().size(); ++e)
        {
            if((*work)[e])
                weight += G.edges()[e].weight;
        }

        if(weight > *bestWeight)
        {
            *bestWeight = weight;
            *bestSolution = *work;
        }

        return;
    }

    if(i >= problematic.size())
        return;

    // Should we take edge i or not?
    // First try with the edge
    trySolution(G, problematic, i+1, work, bestSolution, bestWeight);

    // and now without.
    (*work)[problematic[i]] = false;
    trySolution(G, problematic, i+1, work, bestSolution, bestWeight);

    // restore previous state
    (*work)[problematic[i]] = true;
}

void postprocessWithDepth(std::vector<Detection>* detections, const cv::Mat_<uint8_t>& segmentation, const cv::Mat_<float>& confidence, const std::vector<std::string>& classes, at::Tensor& cloud, const std::vector<Eigen::Vector3f>& objectSizes)
{
    constexpr float LOOK_DIST = 50; // px

    // Compute 3D centroids
    std::vector<std::size_t> centroidCount(classes.size(), 0);
    std::vector<Detection*> classToDetection(classes.size(), 0);
    std::vector<Eigen::Vector3f> graspPoint(classes.size(), Eigen::Vector3f::Zero());

    auto cloudAcc = cloud.accessor<float, 3>();

    for(std::size_t i = 0; i < classes.size(); ++i)
    {
        auto it = std::find_if(detections->begin(), detections->end(), [&](const Detection& a){
            return a.object == classes[i];
        });
        if(it != detections->end())
        {
            Detection* det = &*it;
            classToDetection[i] = det;
            auto cloudPoint = cloudAcc[det->suctionPoint.y][det->suctionPoint.x];
            graspPoint[i] = {cloudPoint[0], cloudPoint[1], cloudPoint[2]};
        }
    }

    for(int y = 0; y < segmentation.rows; ++y)
    {
        for(int x = 0; x < segmentation.cols; ++x)
        {
            unsigned int segmCode = segmentation(y,x);
            assert(segmCode > 0);
            assert(segmCode <= classes.size());

            Detection* det = classToDetection[segmCode];
            if(det)
            {
                auto cloudPoint = cloudAcc[det->suctionPoint.y][det->suctionPoint.x];
                det->cloudCentroid += Eigen::Vector3f::Map(&cloudPoint[0]);
                centroidCount[segmCode - 1]++;
            }
        }
    }
    for(std::size_t i = 0; i < classes.size(); ++i)
    {
        if(classToDetection[i])
            classToDetection[i]->cloudCentroid /= centroidCount[i];
    }

    graph::VertexID boxClass = -1;
    {
        auto it = std::find(classes.begin(), classes.end(), "box");
        if(it != classes.end())
            boxClass = it - classes.begin();
        else
        {
            fprintf(stderr, "Need box class in set of classes!");
            return;
        }
    }

    graph::VertexID unknownClass = -1;
    {
        auto it = std::find(classes.begin(), classes.end(), "unknown");
        if(it != classes.end())
            unknownClass = it - classes.begin();
    }

    // Create a directed graph containing all objects.
    // The graph will represent the detected relationships (e.g. A is on top of B).
    graph::Graph G(classes.size());

    // All items are above the box.
    for(graph::VertexID v = 0; v < classes.size(); ++v)
    {
        if(v != boxClass)
            G.addEdge(v, boxClass, 1);
    }

    for(std::size_t detectionIdx = 0; detectionIdx < detections->size(); ++detectionIdx)
    {
        auto& detection = (*detections)[detectionIdx];

        graph::VertexID myClass = -1;
        {
            auto it = std::find(classes.begin(), classes.end(), detection.object);
            if(it != classes.end())
                myClass = it - classes.begin();
        }

        // Go along the outer contour
        const auto& contour = detection.contours.at(0);

        unsigned int countAbove = 0;
        unsigned int countBelow = 0;
        double sumBelow = 0;
        unsigned int outOfBounds = 0;
        unsigned int boxHit = 0;

        Eigen::VectorXi countBelowPerClass = Eigen::VectorXi::Zero(classes.size());
        Eigen::VectorXd sumBelowPerClass = Eigen::VectorXd::Zero(classes.size());

        for(std::size_t i = 0; i < contour.size()-1; ++i)
        {
            auto& p1 = contour[i];
            auto& p2 = contour[i+1];

            // The outer contour always goes counterclockwise.
            // => The outer side is on the right side of the line from p1 to p2.

            cv::Point diff = p2 - p1;

            Eigen::Vector2f normal(-diff.y, diff.x);

            Eigen::Vector2f dist = LOOK_DIST * normal.normalized();

            cv::Point myPoint(p1 - cv::Point(dist.x(), dist.y()));
            cv::Point theirPoint(p1 + cv::Point(dist.x(), dist.y()));

            if(theirPoint.x < 0 || theirPoint.y < 0 || theirPoint.x >= segmentation.cols || theirPoint.y >= segmentation.rows
                || myPoint.x < 0 || myPoint.y < 0 || myPoint.x >= segmentation.cols || myPoint.y >= segmentation.rows)
            {
                countAbove++;
                outOfBounds++;
                continue;
            }

            uint8_t segmCode = segmentation(theirPoint.y, theirPoint.x);

            // LUA is 1-based
            if(unknownClass != static_cast<graph::VertexID>(-1) && segmCode == unknownClass + 1)
                continue;

            if(segmCode == boxClass)
            {
                boxHit++;
                continue;
            }

            if(myClass != static_cast<graph::VertexID>(-1) && segmCode == myClass)
                continue;

            float myConfidence = confidence(myPoint.y, myPoint.x);
            float theirConfidence = confidence(theirPoint.y, theirPoint.x);
            if(theirConfidence < LOW_SEGMENTATION_THRESHOLD)
                continue;

            // Look up the corresponding detection if there is any
            const Detection* theirDetection = 0;
            for(auto& det : *detections)
            {
                if(det.object == classes[segmCode])
                {
                    theirDetection = &det;
                    break;
                }
            }
            if(!theirDetection)
                continue;

            // Find furthest point on their contour
            double furthestDist = 0.0;
            for(auto& point : theirDetection->contours.at(0))
            {
                cv::Point diff = point - theirPoint;
                double d = diff.x*diff.x + diff.y*diff.y;
                if(d > furthestDist)
                    furthestDist = d;
            }
            furthestDist = std::sqrt(furthestDist);

            double diagonal = objectSizes[theirDetection->classID].norm();
            double pixelDiagonal = (std::max<double>(cloud.size(0), cloud.size(1)) / TOTE_WIDTH) * diagonal;

            if(furthestDist > 1.2 * pixelDiagonal)
                continue;

            float myDepth = cloudAcc[myPoint.y][myPoint.x][2];
            float depth = cloudAcc[theirPoint.y][theirPoint.x][2];

            if(!std::isfinite(myDepth) || !std::isfinite(depth))
                continue;

            double centroidPrior = 1.0;
//             if(detection.object == "hanes_socks" && classes[segmCode-1] == "reynolds_wrap" && depth <= myDepth)
//             {
//                 ROS_INFO("myDepth: %f, depth: %f, my graspPoint.z: %f", myDepth, depth, graspPoint[myClass].z());
//             }

            if(depth - graspPoint[myClass].z() > 0.02)
                centroidPrior = 0.1;

//             printf("depth: %f, myDepth: %f\n", depth, myDepth);
            if(depth > myDepth)
            {
                // The other object is further downwards
                countAbove++;
            }
            else
            {
                // The other object is further above
                countBelow++;
                sumBelow += centroidPrior * theirConfidence * myConfidence;

                if(segmCode > 0 && segmCode <= classes.size())
                {
                    countBelowPerClass[segmCode]++;
                    sumBelowPerClass[segmCode] += centroidPrior * theirConfidence * myConfidence;
                }
            }
        }

//         printf("Object %s is %u above and %u below (%u box, %u oob). Per class:\n", detection.object.c_str(), countAbove, countBelow, boxHit, outOfBounds);
        for(std::size_t i = 0; i < classes.size(); ++i)
        {
            if(countBelowPerClass[i] < 10)
                continue;

            if(i == unknownClass || i == boxClass)
                continue;

            if(myClass == static_cast<graph::VertexID>(-1) || i == myClass)
                continue;

            G.addEdge(i, myClass, sumBelowPerClass[i]);
//             printf(" - %s is %f above\n", classes[i].c_str(), sumBelowPerClass[i]);
        }

        detection.occlusionFactor = static_cast<double>(countAbove) / (countAbove + countBelow);
        detection.aboveBelow = countAbove - countBelow;
    }

    // Filter direct reciprocal edges
    {
        std::vector<graph::Edge> newEdges;
        newEdges.reserve(G.edges().size());
        std::vector<bool> visited(G.edges().size(), false);

        for(graph::EdgeID e = 0; e < G.edges().size(); ++e)
        {
            if(visited[e])
                continue;

            visited[e] = true;

            auto& edge = G.edges()[e];
            auto w = G.vertices()[edge.to];

            graph::EdgeID otherIdx = graph::UNSET;
            for(auto m : w.successors)
            {
                auto& otherEdge = G.edges()[m];
                if(otherEdge.to == edge.from)
                {
                    otherIdx = m;
                    break;
                }
            }

            if(otherIdx == graph::UNSET)
            {
                // Simple edge
                newEdges.push_back(edge);
            }
            else
            {
                visited[otherIdx] = true;

                // Reciprocal edge
                auto& otherEdge = G.edges()[otherIdx];
                if(edge.weight > otherEdge.weight)
                    newEdges.emplace_back(edge.from, edge.to, edge.weight - otherEdge.weight);
                else
                    newEdges.emplace_back(edge.to, edge.from, otherEdge.weight - edge.weight);
            }
        }

        G.edges() = std::move(newEdges);
        G.reconstructSuccessors();
    }

    // Find strongly connected components (= components containing cycles or singletons!)
    auto components = graph::tarjan(G);

    printf("%lu strongly connected components (%lu vertices)\n", components.size(), G.vertices().size());

//     for(auto& component : components)
//     {
//         printf(" -");
//         for(auto& v : component)
//             printf(" %d", v);
//         printf("\n");
//     }

    // Figure out edges belonging to non-singleton components
    // These may belong to cycles.
    std::vector<graph::EdgeID> problematicEdges;
    {
        for(auto& component : components)
        {
            if(component.size() <= 1)
                continue;

            printf("Found component with %lu vertices\n", component.size());

            std::vector<bool> inComponent(G.vertices().size(), false);
            for(auto v : component)
                inComponent[v] = true;

            for(auto v : component)
            {
                printf("Node %u\n", v);
                for(auto edgeID : G.vertices()[v].successors)
                {
                    auto& edge = G.edges()[edgeID];
                    printf("%u: %u -> %u\n", edgeID, edge.from, edge.to);
                    if(inComponent[G.edges()[edgeID].to])
                        problematicEdges.push_back(edgeID);
                }
            }
        }
    }

    {
        graph::Graph problematicG = G;
        problematicG.edges().clear();
        for(auto& edgeID : problematicEdges)
            problematicG.edges().push_back(G.edges()[edgeID]);
        problematicG.reconstructSuccessors();
        std::ofstream file("G_prob.dot");
        problematicG.toDot(file, classes, [&](const graph::VertexID v){
            return true;
        });
    }

    printf("Found %lu problematic edges\n", problematicEdges.size());
    std::sort(problematicEdges.begin(), problematicEdges.end(), [&](graph::EdgeID a, graph::EdgeID b) {
        return G.edges()[a].weight > G.edges()[b].weight;
    });
    for(auto& e : problematicEdges)
        printf(" - %20s -> %20s (%f)\n", classes[G.edges()[e].from].c_str(), classes[G.edges()[e].to].c_str(), G.edges()[e].weight);


    /*if(problematicEdges.size() > 32)
    {
        ROS_ERROR("I do not know how to solve for so many problematic edges.");
        ROS_ERROR("Falling back to stupid approximation");

        std::vector<graph::Edge> set1;
        std::vector<graph::Edge> set2;

        for(const auto& edge : G.edges())
        {
            if(edge.from < edge.to)
                set1.push_back(edge);
            else
                set2.push_back(edge);
        }

        // set1 and set2 are both acyclic, see which one is better
        double weight1 = std::accumulate(set1.begin(), set1.end(), 0.0, [](double w, const graph::Edge& e){
            return w + e.weight;
        });
        double weight2 = std::accumulate(set2.begin(), set2.end(), 0.0, [](double w, const graph::Edge& e){
            return w + e.weight;
        });

        if(weight1 > weight2)
            G.edges() = set1;
        else
            G.edges() = set2;
        G.reconstructSuccessors();
    }
    else */if(problematicEdges.size() != 0)
    {
        // FIXME: This could be done per strongly connected component

        std::vector<bool> work(G.edges().size(), true);

        std::size_t cutoff = std::min<std::size_t>(problematicEdges.size(), 20);
        for(std::size_t i = cutoff; i < problematicEdges.size(); ++i)
        {
            work[problematicEdges[i]] = false;
        }
        problematicEdges.resize(cutoff);

        std::vector<bool> bestSolution = work;
        double bestWeight = 0.0;
        trySolution(G, problematicEdges, 0, &work, &bestSolution, &bestWeight);

        printf("Best solution has weight %f\n", bestWeight);

        std::vector<graph::Edge> edges;
        edges.reserve(G.edges().size());

        for(auto& edgeID : problematicEdges)
        {
            if(!bestSolution[edgeID])
                continue;

            auto edge = G.edges()[edgeID];
            printf("Solution includes %s -> %s (%f)\n",
                classes[edge.from].c_str(),
                classes[edge.to].c_str(),
                edge.weight
            );
        }

        for(graph::EdgeID e = 0; e < G.edges().size(); ++e)
        {
            if(bestSolution[e])
            {
                auto edge = G.edges()[e];
                edges.push_back(edge);
            }
        }

        G.edges() = edges;
        G.reconstructSuccessors();
    }

    for(auto& detection : *detections)
    {
        unsigned int myIdx = -1;
        {
            auto it = std::find(classes.begin(), classes.end(), detection.object);
            if(it != classes.end())
                myIdx = it - classes.begin();
            else
                continue;
        }

        for(auto edge : G.edges())
        {
            if(edge.to != myIdx)
                continue;

            if(edge.from == boxClass || edge.to == boxClass)
                continue;
            if(edge.from == unknownClass || edge.to == unknownClass)
                continue;

            detection.objectsAbove.push_back(classes[edge.from]);
            detection.objectsAboveWeight.push_back(edge.weight);
        }
    }

    for(auto& detection : *detections)
    {
        unsigned int myIdx = -1;
        {
            auto it = std::find(classes.begin(), classes.end(), detection.object);
            if(it != classes.end())
                myIdx = it - classes.begin();
            else
                continue;
        }

        std::vector<bool> visited(G.vertices().size(), false);

        std::stack<graph::VertexID> next;
        next.push(myIdx);
        visited[myIdx] = true;

        if(boxClass != static_cast<graph::VertexID>(-1))
            visited[boxClass] = true;
        if(unknownClass != static_cast<graph::VertexID>(-1))
            visited[unknownClass] = true;

        while(!next.empty())
        {
            graph::VertexID v = next.top();
            next.pop();

            for(auto& edgeID : G.vertices()[v].predecessors)
            {
                graph::VertexID w = G.edges()[edgeID].from;

                if(visited[w])
                    continue;

                visited[w] = true;
                detection.totalObjectsAbove++;
                next.push(w);
            }
        }
    }
}

cv::Mat_<cv::Vec3b> visualizeDetections(const cv::Mat_<cv::Vec3b>& rgb, const std::vector<Detection>& detectionsIn, bool grasps)
{
//     std::vector<Detection> detections = detectionsIn;
//     std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
//         return static_cast<float>(a.centroid.y) / a.centroid.x < static_cast<float>(b.centroid.y) / b.centroid.x;
//     });

    std::vector<Detection> leftDetections;
    std::vector<Detection> rightDetections;

    for(auto& det : detectionsIn)
    {
        if(det.centroid.x < rgb.cols/2)
            leftDetections.push_back(det);
        else
            rightDetections.push_back(det);
    }

    cv::Mat_<cv::Vec3b> vis(rgb.rows, rgb.cols + 2000);
    vis = cv::Vec3b(255, 255, 255);
    std::cout << cv::Rect(cv::Point(1000, 0), rgb.size()) << std::endl;
    rgb.copyTo(vis(cv::Rect(cv::Point(1000, 0), rgb.size())));

    const double FONT_SCALE = 1.0;
    // Render left side
    int i = 0;
    {
        int y = 0;
        const int skip = 30;

        while(!leftDetections.empty())
        {
            cv::Point start(800, y + (3*skip)/2);
            auto it = std::min_element(leftDetections.begin(), leftDetections.end(), [&](const Detection& a, const Detection& b) {
                cv::Point2d diff_a = a.centroid + cv::Point(1000, 0) - start;
                cv::Point2d diff_b = b.centroid + cv::Point(1000, 0)- start;

                return diff_a.y/diff_a.x < diff_b.y/diff_b.x;
            });

            Detection det = *it;
            leftDetections.erase(it);

            cv::Scalar color = COLORS[i % COLORS.size()];

            int baseline;
            cv::Size size = cv::getTextSize(det.object.c_str(), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, 4, &baseline);
            cv::putText(vis, det.object.c_str(), cv::Point(800 - size.width, y + skip), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, cv::Scalar(0, 0, 0), 4);

            char buf[256];

            snprintf(buf, sizeof(buf), "conf: %f", det.confidence);
            size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, 4, &baseline);

            cv::putText(vis, buf, cv::Point(800 - size.width, y + 2*skip), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, cv::Scalar(0, 0, 0), 4);

//             snprintf(buf, sizeof(buf), "area: %f", det.visibleAreaFactor);
//             cv::putText(vis, buf, cv::Point(0, y + 3*skip), cv::FONT_HERSHEY_SIMPLEX, 2, color, 4);

            cv::line(vis, start, det.centroid + cv::Point(1000, 0), color, 2);

            cv::drawContours(vis(cv::Rect(cv::Point(1000, 0), rgb.size())), det.contours, -1, color, 4);

            if(grasps)
            {
                cv::circle(vis, det.suctionPoint + cv::Point(1000, 0), 10, color, -1);
                cv::circle(vis, det.polygonCentroid + cv::Point(1000, 0), 5, color, -1);
            }

            y += 2.5*skip;
            ++i;
        }
    }

    // Render right side
    {
        int y = 0;
        const int skip = 30;
        while(!rightDetections.empty())
        {
            cv::Point start(vis.cols - 1000 + 200, y + (3*skip)/2);
            auto it = std::max_element(rightDetections.begin(), rightDetections.end(), [&](const Detection& a, const Detection& b) {
                cv::Point2d diff_a = a.centroid + cv::Point(1000, 0) - start;
                cv::Point2d diff_b = b.centroid + cv::Point(1000, 0) - start;

                return diff_a.y/diff_a.x < diff_b.y/diff_b.x;
            });

            Detection det = *it;
            rightDetections.erase(it);

            cv::Scalar color = COLORS[i % COLORS.size()];

            cv::putText(vis, det.object.c_str(), cv::Point(vis.cols - 1000 + 200, y + skip), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, cv::Scalar(0, 0, 0), 4);

            char buf[256];

            snprintf(buf, sizeof(buf), "conf: %f", det.confidence);
            cv::putText(vis, buf, cv::Point(vis.cols - 1000 + 200, y + 2*skip), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, cv::Scalar(0, 0, 0), 4);

//             snprintf(buf, sizeof(buf), "area: %f", det.visibleAreaFactor);
//             cv::putText(vis, buf, cv::Point(vis.cols - 1000, y + 3*skip), cv::FONT_HERSHEY_SIMPLEX, 2, color, 4);

            cv::line(vis, start, det.centroid + cv::Point(1000, 0), color, 2);

            cv::drawContours(vis(cv::Rect(cv::Point(1000, 0), rgb.size())), det.contours, -1, color, 4);

            if(grasps)
            {
                cv::circle(vis, det.suctionPoint + cv::Point(1000, 0), 10, color, -1);
                cv::circle(vis, det.polygonCentroid + cv::Point(1000, 0), 5, color, -1);
            }

            y += 2.5*skip;
            ++i;
        }
    }

    return vis;
}

void processDetections(std::vector<Detection>& detections)
{
    printf("Items from perception:\n");
    for(auto& d : detections)
    {
        printf("  item '%20s' confidence=%.3f visible_surface_factor=%.3f above=%3d\n",
            d.object.c_str(), d.confidence, d.visibleAreaFactor, d.totalObjectsAbove);
    }

    // Confidence check doesn't make sense without real perception
#if 0
    std::sort(detections.begin(), detections.end(), [](const auto& a, const auto& b){
        return a.confidence > b.confidence;
    });

    std::size_t upperLimit = std::max<std::size_t>(6, 3*detections.size() / 4);
    detections.resize(std::min(upperLimit, detections.size()));

    printf("Surviving items after confidence check:\n");
    for(auto& d : detections)
        printf("  item '%20s' confidence=%.3f visible_surface_factor=%.3f above=%3d\n",
            d.object.c_str(), d.confidence, d.visibleAreaFactor, d.totalObjectsAbove);
#endif

    std::sort(detections.begin(), detections.end(), [](const auto& a, const auto& b){
//         return a.visible_surface_factor > b.visible_surface_factor;

        std::size_t aboveA = a.totalObjectsAbove;
        std::size_t aboveB = b.totalObjectsAbove;

        if(aboveA == aboveB)
            return a.visibleAreaFactor > b.visibleAreaFactor;
        else
            return aboveA < aboveB;
    });
    detections.resize(std::min<std::size_t>(detections.size(), std::max<std::size_t>(3, detections.size() / 2)));

    printf("Surviving items after area check:\n");
    for(auto& d : detections)
    {
        printf("  item '%20s' confidence=%.3f visible_surface_factor=%.3f above=%3d\n",
            d.object.c_str(), d.confidence, d.visibleAreaFactor, d.totalObjectsAbove);
    }
}

}

// Python bindings

PYBIND11_MAKE_OPAQUE(std::vector<arc_perception::Detection>);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<arc_perception::Detection>(m, "Detection")
        .def("__str__", [&](arc_perception::Detection& det){
            return "Detection('" + det.object + "')";
        })
        .def("__repr__", [&](arc_perception::Detection& det){
            return "Detection('" + det.object + "')";
        })
        .def_property_readonly("name", [](const arc_perception::Detection& det){
            return det.object;
        })
        .def_property_readonly("suction_point", [](const arc_perception::Detection& det){
            return torch::tensor({det.suctionPoint.x, det.suctionPoint.y});
        })
    ;
    py::bind_vector<std::vector<arc_perception::Detection>>(m, "DetectionList");

    m.def("postprocess_segmentation", [](
        at::Tensor& segmentation, at::Tensor& confidence,
        const std::vector<std::string>& classes, at::Tensor& objectSizes, std::vector<double>& objectWeights)
    {
        auto cpuSegm = segmentation.to(at::kByte).cpu().contiguous();
        if(cpuSegm.dim() != 2)
            throw std::invalid_argument{"segmentation tensor must be 2D"};

        cv::Mat_<uint8_t> cvSegm(cpuSegm.size(0), cpuSegm.size(1), cpuSegm.data_ptr<uint8_t>());

        auto cpuConf = confidence.to(at::kFloat).cpu().contiguous();
        if(cpuConf.dim() != 2)
            throw std::invalid_argument{"confidence tensor must be 2D"};

        cv::Mat_<float> cvConf(cpuConf.size(0), cpuConf.size(1), cpuConf.data_ptr<float>());


        std::vector<Eigen::Vector3f> cvObjectSizes;
        auto sizeAcc = objectSizes.accessor<float, 2>();
        for(int i = 0; i < objectSizes.size(0); ++i)
            cvObjectSizes.emplace_back(sizeAcc[i][0], sizeAcc[i][1], sizeAcc[i][2]);

        return arc_perception::postprocessSegmentation(
            cvSegm, cvConf, classes, cvObjectSizes, objectWeights
        );
    },
        py::arg("segmentation"), py::arg("confidence"), py::arg("classes"),
        py::arg("object_sizes"), py::arg("object_weights"));

    m.def("postprocess_with_depth", [&](
        std::vector<arc_perception::Detection>* detections,
        at::Tensor& segmentation, at::Tensor& confidence,
        const std::vector<std::string>& classes,
        at::Tensor& cloud,
        at::Tensor& objectSizes)
    {
        auto cpuSegm = segmentation.to(at::kByte).cpu().contiguous();
        if(cpuSegm.dim() != 2)
            throw std::invalid_argument{"segmentation tensor must be 2D"};

        cv::Mat_<uint8_t> cvSegm(cpuSegm.size(0), cpuSegm.size(1), cpuSegm.data_ptr<uint8_t>());

        auto cpuConf = confidence.to(at::kFloat).cpu().contiguous();
        if(cpuConf.dim() != 2)
            throw std::invalid_argument{"confidence tensor must be 2D"};

        cv::Mat_<float> cvConf(cpuConf.size(0), cpuConf.size(1), cpuConf.data_ptr<float>());

        std::vector<Eigen::Vector3f> cvObjectSizes;
        auto sizeAcc = objectSizes.accessor<float, 2>();
        for(int i = 0; i < objectSizes.size(0); ++i)
            cvObjectSizes.emplace_back(sizeAcc[i][0], sizeAcc[i][1], sizeAcc[i][2]);

        at::Tensor cloudCPU = cloud.cpu();
        arc_perception::postprocessWithDepth(
            detections, cvSegm, cvConf, classes, cloudCPU, cvObjectSizes
        );
    },
        py::arg("detections"),
        py::arg("segmentation"), py::arg("confidence"), py::arg("classes"),
        py::arg("cloud"), py::arg("object_sizes"));

    m.def("process_detections", &arc_perception::processDetections,
        py::arg("detections"));

    m.def("visualize_detections", [&](
        at::Tensor& rgb,
        const std::vector<arc_perception::Detection>& detectionsIn,
        bool grasps)
    {
        auto cpuRGB = rgb.to(at::kByte).cpu().contiguous();
        if(cpuRGB.dim() != 3 || cpuRGB.size(2) != 3)
            throw std::invalid_argument{"RGB tensor must be 3D"};

        cv::Mat_<cv::Vec3b> cvRGB(cpuRGB.size(0), cpuRGB.size(1),
            reinterpret_cast<cv::Vec3b*>(cpuRGB.data_ptr<uint8_t>())
        );

        cv::Mat_<cv::Vec3b> vis = arc_perception::visualizeDetections(
            cvRGB, detectionsIn, grasps
        );

        return torch::from_blob(
            reinterpret_cast<uint8_t*>(vis.data),
            {vis.rows, vis.cols, 3},
            at::kByte
        ).clone();
    },
        py::arg("rgb"), py::arg("detections"), py::arg("grasps")=true);

}

