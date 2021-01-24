#include <iostream>
#include "bh.h"
#include "Kmeans.h"
#include "MinHash.hpp"
#include <armadillo>
#include <set>

using namespace std;
using namespace arma;

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void writeToFile(string filename, map<int, string> filterMap) {
    //Write bloom filters into file
    ofstream stream(filename);
    cout << "Writing filters" << endl;
    for (auto filter : filterMap) {
        stream << to_string(filter.first) << filter.second << '\n';
        // Add '\n' character  ^^^^
    }
    stream << '\n';
    stream.flush();
}

/**
 * Seperate bloom filters from clusters given the prediction for corresponding data point
 * @param data Matrix of bloom filters
 * @param pred Column matrix of cluster prediction for associated data point
 * @param clusterCount No of clusters
 * @param outfilePrefix Save file name prefix (Without extension)
 */
void seperateClusters(Mat<float> &data, Mat<short> pred, int clusterCount, string outfilePrefix) {
    for(int i = 0; i < clusterCount; i++) {
        //Filter indices of filters belonging to cluster
        Col<uword> indices = find(pred == i);
        //Filter out cluster data
        Mat<short> clusterData = conv_to<Mat<short>>::from(data.rows(indices));
        //Write to file
        string outfile = outfilePrefix + to_string(i) +".txt";
        clusterData.save(outfile, arma::csv_ascii);
    }
}

string replace(string str, string old, string replacement) {
    size_t index = 0;
    while (true) {
        /* Locate the substring to replace. */
        index = str.find(old, index);
        if (index == std::string::npos) {
            return str;
        }

        /* Make the replacement. */
        str.replace(index, 1, replacement);

        /* Advance index forward so the next iteration doesn't pick it up as well. */
        index += 2;
    }
}

map<unsigned long, set<string>> combineLocalBuckets(vector<map<unsigned long, vector<string>>> totalWorkerBuckets) {
    map<unsigned long, set<string>> combinedBuckets;
    for (auto workerBuckets: totalWorkerBuckets) {
        for (auto bucket: workerBuckets) {
            unsigned long bucketID = bucket.first;
            vector<string> clusters = bucket.second;
            set<string> existingBucket = combinedBuckets[bucketID];
            copy(clusters.begin(), clusters.end(), std::inserter(existingBucket, existingBucket.end()));
        }
    }

    return combinedBuckets;
}

/**
 * Method call for party coordinator to combine buckets of clusters given by each party to compute the similar clusters
 * Clusters that fall under the same bucket will be considered similar
 * Buckets that have clusters from all the parties will be kept since only they correspond to the possible common entities
 * across all parties
 * @param allBuckets Map of party IDs to mapping of bucket IDs to sets of clusters of that party
 * @return Map of bucket ID to clusters across all parties
 */
map<unsigned long, map<string, set<string>>> getSimilarClusters(map<string, map<unsigned long, set<string>>> allBuckets) {
    map<unsigned long, map<string, set<string>>> combinedBuckets; //Map of bucket to organisation-cluster
    //Combine all organization buckets
    for (auto orgBuckets: allBuckets) {
        string partyID = orgBuckets.first;

        for (auto bucket: orgBuckets.second) {
            unsigned long bucketID = bucket.first;
            set<string> clusters = bucket.second;
            combinedBuckets[bucketID][partyID].insert(clusters.begin(), clusters.end());
        }
    }

    //Filter buckets
    map<unsigned long, map<string, set<string>>> filteredBuckets;

    for (auto bucket: combinedBuckets) {
        if (bucket.second.size() >= 3) {
            filteredBuckets[bucket.first] = bucket.second;
        }
    }

    return filteredBuckets;
}

/**
 * Compare filters against each other and get the most similar.
 * Classify as similar or not using a similarity threshold
 * @param selfFilters Filter coming from the party doing the computation
 * @param otherFilters Filter from the other party
 * @param similarityThreshold Similarity threshold for classification
 * @return Vector of two maps
 */
vector<map<string, string>> compareFilters(Mat<short> &selfFilters, Mat<short> &otherFilters, float similarityThreshold = 0.9) {
    map<string, string> commonEntityMapSelf;
    map<string, string> commonEntityMapOther;

    Row<short> denominator = arma::sum(selfFilters, 0) + arma::sum(otherFilters, 0); //denominator of dice coeff

    //For each filter in self cluster, compare against filters from other clusters and determine most similar filter
    for (int i = 0; i < selfFilters.n_cols; i++) {
        //Compute dice coefficient values
        Col<short> selfFilter = selfFilters.col(i);
        Row<short> numerator = 2 * arma::sum(otherFilters.each_col() % selfFilter, 0); //Numerator to compute the dice coeff
        Row<float> diceCoeff = (conv_to<Mat<float>>::from(numerator) / conv_to<Mat<float>>::from(denominator));

        //Get the most similar filter and check if it's meets the similarity threshold
        uword maxIndex = arma::index_max(diceCoeff); //arg max
        if (diceCoeff(maxIndex) > similarityThreshold) {
            //Assign the two filters as the same common entity
            commonEntityMapSelf[to_string(i)] = to_string(maxIndex);
            commonEntityMapOther[to_string(maxIndex)] = to_string(i);
        }

    }

    return {commonEntityMapSelf, commonEntityMapOther};
}

void combineFilterwiseResults(vector<map<string, string>> results) {
    map<string, string> combinedEntityMap;
    for (auto filterEntityMap: results) {
        for (auto entity: filterEntityMap) {
            combinedEntityMap[entity.first] = entity.second;
        }
    }
}

/**
 * Compute the common entities accross all parties given a chainable pairwise common entity information
 * @param pairwiseCommonEntities Map of party-ids to the mappings of common entities between other parties
 */
map<string, vector<string>> synchronizeCommonEntities(map<string, map<string, map<string, string>>> pairwiseCommonEntities) {
    map<string, vector<string>> partyCommonEntityMap;
    string firstPassEndParty;

    //Get common entitiy ids of self
    string currentParty = "A"; //Self party ID
    string nextParty = pairwiseCommonEntities[currentParty].begin() -> first;
    vector<string> currentPartyIds;
    for (auto idPairs: pairwiseCommonEntities[currentParty][nextParty]) {
        currentPartyIds.emplace_back(idPairs.first);
    }

    //First pass iteration through intermediate pairwise results
    for(int i = 0; i < pairwiseCommonEntities.size(); i++) {
        auto commonEntityMap = pairwiseCommonEntities[currentParty][nextParty];
        //Iterate through common entity ids of currentParty with nextParty
        vector<string> nextPartyIds;
        for (string id: currentPartyIds) {
            partyCommonEntityMap[currentParty].emplace_back(id);
            //If id is present in the next party common entities, mark it to check in the next iterationa
            if (commonEntityMap.find(id) != commonEntityMap.end()) {
                nextPartyIds.emplace_back(pairwiseCommonEntities[currentParty][nextParty][id]);
            }
        }
        //Party to iterate
        currentParty = nextParty;
        //Filtered next set of ids, when the loop terminates we will have entities of self which are common for all parties
        currentPartyIds = nextPartyIds;
        //Change pointer to next paty in the chain
        nextParty = pairwiseCommonEntities[currentParty].begin() -> first;
    }

    //Second pass to compute common entities across all entities
    for(int i = 0; i < pairwiseCommonEntities.size(); i++) {
        //Replace with the entities of current party which are common for all parties
        partyCommonEntityMap[currentParty] = currentPartyIds;

        nextParty = pairwiseCommonEntities[currentParty].begin() -> first;
        vector<string> nextPartyIds;
        //For only the entities common for all, get mapping entities of next party
        for (string id: currentPartyIds) {
            nextPartyIds.emplace_back(pairwiseCommonEntities[currentParty][nextParty][id]);
        }
        //Party to iterate
        currentParty = nextParty;
        //Filtered next set of ids
        currentPartyIds = nextPartyIds;
        //Change pointer to next paty in the chain
        nextParty = pairwiseCommonEntities[currentParty].begin() -> first;
    }

    return partyCommonEntityMap;
}


int main() {
    map<int, vector<string>> entityData;
    map<int, vector<int>> neighborhoodData;
    entityData[0] = {"John", "Doe", "24"};
    entityData[1] = {"Jane", "Dawson", "24"};

    //Read attributes from file
    std::ifstream dataFile;
    cout << "reading file" << endl;
    dataFile.open("/root/CLionProjects/EntityResolution/entityData.txt", std::ios::binary | std::ios::in);

    char splitter;
    string line;

    std::getline(dataFile, line);

    splitter = ' ';

    cout << "Getting data" << endl;
    while (!line.empty()) {

        int strpos = line.find(splitter);
        string nodeID = line.substr(0, strpos);
        string weakIDStr = line.substr(strpos + 1, -1);
        entityData[stoi(nodeID)] = split(weakIDStr, splitter);
        cout << line << endl;


        std::getline(dataFile, line);
        while (!line.empty() && line.find_first_not_of(splitter) == std::string::npos) {
            std::getline(dataFile, line);
        }
    }
    dataFile.close();

    //Read edgelist from file
    std::ifstream edgeFile;
    cout << "reading file" << endl;
    edgeFile.open("/root/CLionProjects/EntityResolution/edgelist.txt");

    std::getline(edgeFile, line);

    splitter = ' ';

    cout << "Getting neighbourhood data" << endl;
    while (!line.empty()) {

        int strpos = line.find(splitter);
        string vertex1 = line.substr(0, strpos);
        string vertex2 = line.substr(strpos + 1, -1);
        if (neighborhoodData.size() <= 5) {
            neighborhoodData[stoi(vertex1)].emplace_back(stoi(vertex2));
        }
        cout << line << endl;


        std::getline(edgeFile, line);
        while (!line.empty() && line.find_first_not_of(splitter) == std::string::npos) {
            std::getline(edgeFile, line);
        }
    }
    edgeFile.close();

    //Create bloom filters
    cout << "Creating filters" << endl;
    map<int, string> attrFilters;
    map<int, string> structFilters;
    //For each entity create attr and structural bloom filters
    int filterSize = 256;
    for (const auto& entity: entityData) {
        //Create attr bloom filter
        BloomFilter attrFilter(filterSize, 4);
        vector<string> attributes = entity.second;
        //Add node attributes to bloom filter
        for (auto attr: attributes) {
            cout << attr << endl;
            attrFilter.insert(attr);
        }
        //Convert bloom filter to appropriate string
        string filterStr = attrFilter.m_bits.to_string();
        cout << filterStr << endl;
        filterStr = replace(filterStr, "0", ",0");
        filterStr = replace(filterStr, "1", ",1");
        attrFilters[entity.first] = filterStr;
        cout << "Attr Filter created " << filterStr << endl;

        //Create structural filter
        BloomFilter structFilter(256, 4);
        //For each neighbour add selected attribute to bloom filter
        for (auto neighbour: neighborhoodData[entity.first]) {
            string selectedAttr = entityData[neighbour][0];
            structFilter.insert(selectedAttr);
        }
        //Convert bloom filter to appropriate string
        filterStr = structFilter.m_bits.to_string();
        cout << filterStr << endl;
        filterStr = replace(filterStr, "0", ",0");
        filterStr = replace(filterStr, "1", ",1");
        structFilters[entity.first] = filterStr;
        cout << "Structural Filter created " << filterStr << endl;
    }

    //Write attr and struct filters separately into files
    writeToFile("attrfilters.txt", attrFilters);
    writeToFile("structfilters.txt", structFilters);


    //Read from bloom filter file(s)
    Mat<float> data;
    data.load("/root/CLionProjects/EntityResolution/attrfilters.txt", arma::csv_ascii);

    //Prepare data
    Mat<float> ids = data.col(0);
    data.shed_col(0);
    inplace_trans(data, "lowmem");

    //Train kmeans clustering for 16 clusters
    int noClusters = 3;
    Kmeans<float> model(3);
    model.fit(data, 10);

    //Apply clustering to bloom filters
    Mat<short> pred = model.apply(data);

    //Re-transpose data for saving
    inplace_trans(data, "lowmem");
    //Join with graph ids again
    data = join_rows(ids, data);
    //For each cluster write bloom filters of said cluster into a separate file
    int clusterCount = model.getMeans().n_cols;
    //Separate attr filters into clusters
    seperateClusters(data, pred, clusterCount, "attrfilterscluster");
    data.clear();
    //Separate struct filters into clusters
    data.load("/root/CLionProjects/EntityResolution/structfilters.txt", arma::csv_ascii);
    seperateClusters(data, pred, clusterCount, "structfilterscluster");

    //Share cluster data with other workers

    //Create cluster representative vectors
    int minhashSize = 100;
    Mat<short> CRVs(minhashSize, noClusters);
    for (int i = 0; i < noClusters; i++) {
        //Load cluster file into memory
        string filename = "/root/CLionProjects/EntityResolution/cluster"+ to_string(i) +"filters.txt";
        Mat<float> clusterData;
        clusterData.load(filename, arma::csv_ascii);
        //Remove node id column
        clusterData.shed_col(0);
        inplace_trans(clusterData, "lowmem");
        //Create minhash signature of cluster
        MinHash minHash(minhashSize, 256);
        Col<short> crv = minHash.generateCRV(clusterData, 50);
        //Store in matrix
//        cout <<"test" << endl;
        CRVs.col(i) = crv;
    }

    //Generate local candidate sets
    int bandLength = 10;
    std::ostringstream s;
    hash<string> stdhash;
    map<unsigned long, vector<string>> lshBuckets;
    for (int i = 0; i < noClusters; i++) {
        Col<short> crv = CRVs.col(i);
        //Convert crv into a string without spaces
        crv.st().raw_print(s);
        string crvStr = s.str();
        crvStr = crvStr.substr(0, (crvStr.size() > 0) ? (crvStr.size()-1) : 0);
        crvStr.erase(remove(crvStr.begin(), crvStr.end(), ' '), crvStr.end());
        //For each band of the crv string, put into buckets
        for (int j = 0; j < bandLength; j++) {
            //Select appropriate band
            string crvband = crvStr.substr(j*bandLength, (j+1)*bandLength);
            unsigned long bucket = stdhash(crvband);
            string name = "A" + to_string(i); //Party name + cluster id
            lshBuckets[bucket].emplace_back(name);
        }
    }

    for (auto e: lshBuckets) {
        cout << e.first << " ";
        for (auto i: e.second) {
            cout <<  i << " ";
        }
        cout << endl;
    }

    //Share






}


