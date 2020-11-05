#include <iostream>
#include "bh.h"
#include "Kmeans.h"
#include <armadillo>

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
    for (auto entity: entityData) {
        //Create attr bloom filter
        BloomFilter attrFilter(256, 4);
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
        for (auto neighbour: neighborhoodData[entity]) {
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

    //Write bloom filters into file
    ofstream stream("bloomfilters2.txt");
    cout << "Writing filters" << endl;
    for (auto filter : attrFilters) {
        stream << to_string(filter.first) << filter.second << '\n';
        // Add '\n' character  ^^^^
    }
    stream << '\n';
    stream.flush();


    //Read from bloom filter file(s)
    Mat<float> data;
    data.load("/root/CLionProjects/EntityResolution/bloomfilters2.csv", arma::csv_ascii);

    //Prepare data
    Mat<float> ids = data.col(0);
    data.shed_col(0);
    inplace_trans(data, "lowmem");

    //Train kmeans clustering for 16 clusters
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
    for(int i = 0; i < clusterCount; i++) {
        //Filter indices of filters belonging to cluster
        Col<uword> indices = find(pred == i);
        //Filter out cluster data
        Mat<short> clusterData = conv_to<Mat<short>>::from(data.rows(indices));
        //Write to file
        string outfile = "cluster" + to_string(i) +"filters.txt";
        clusterData.save(outfile, arma::csv_ascii);
    }

}


