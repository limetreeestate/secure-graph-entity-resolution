//
// Created by root on 10/10/20.
//

#ifndef ENTITYRESOLUTION_KMEANS_H
#define ENTITYRESOLUTION_KMEANS_H

#include <armadillo>

template <typename T>
class Kmeans {
public:
    Kmeans(uint8_t k) {
        this->k = k;
    }

    arma::Mat<T> getMeans() {
        return means;
    }

    void fit(arma::Mat<T> &data, uint8_t noOfIterations, bool printMode = true) {
        bool status = kmeans(means, data, k, arma::random_spread, 10, true);

        if(status == false) {
            std::cout << "clustering failed" << std::endl;
        }
    }

    void fit(arma::Mat<T> &data, arma::Mat<T> &means, uint8_t noOfIterations, bool printMode = true) {
        bool status = kmeans(means, data, k, arma::keep_existing, 10, true);

        if(status == false) {
            std::cout << "clustering failed" << std::endl;
        }
    }

    arma::Mat<short> apply(arma::Mat<T> &data) {
        arma::Mat<short> predictions(1, data.n_cols);

        for (int i = 0; i < data.n_cols; i++) {
            arma::Col<float> datapoint = data.col(i);

            double best_dist = arma::Datum<double>::inf;
            uint8_t best_g = 0;
            for(uint8_t g=0; g < means.n_cols; ++g)
            {
                const double tmp_dist = sum(square(datapoint - means.col(g)));
                if(tmp_dist <= best_dist) {
                    best_dist = tmp_dist;
                    best_g = g;
                }
            }

            predictions[i] = best_g;
        }

        return predictions;
    }

private:
    uint8_t k;
    arma::Mat<T> means;

};

#endif //ENTITYRESOLUTION_KMEANS_H
