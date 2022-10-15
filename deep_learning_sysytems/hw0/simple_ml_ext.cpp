#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <assert.h>

namespace py = pybind11;

void transpose(float *a, float b[], size_t m, size_t n){
    float new_a[m*n];
    for (unsigned int i = 0; i < m; ++i )
        for (unsigned int j = 0; j < n; ++j )
            new_a[j*m+i] = a[i*n+j];
    for (unsigned int i=0; i<m*n; i++) 
        b[i] = new_a[i];
}

void multiply(float a[], size_t row1, size_t col1, float b[], size_t row2, size_t col2, float *d) 
{
    assert(col1 == row2);
    float c[row1*col2];
    for (unsigned int i=0; i<row1; i++) {
        for (unsigned int j=0; j<col2; j++) {
            float sum=0;
            for (unsigned int k=0; k<col1; k++)
                sum = sum + a[i*col1+k] * b[k * col2 + j];
            c[i*col2+j] = sum;
        }
    }
    for (unsigned int i=0; i<row1*col2; i++) 
        d[i] = c[i];
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    unsigned char I[m*k];
    unsigned int max = ceil(m / batch);

    /// one-hot-encoding
    for(unsigned int i=0; i<m; i++)
        for(unsigned int j=0; j<k; j++)
            I[k*i+j] = (1 << j) & (1 << y[i]) ? 1 : 0;

    /// batch iterations
    for(unsigned int i=0; i<max; i++){
        /*float Z[k*batch];
        float X_batch[n*batch];
        float gradients[k*n];*/

        float *Z;
        float *X_batch;
        float *gradients;
        float *theta_T;

        X_batch = new float[n*batch];
        Z = new float[k*batch];
        gradients = new float[k*n];
        theta_T = new float[k*n];
        
        for(unsigned int j=i*n*batch; j<n*std::min((i+1)*batch, m); j++)
            X_batch[j] = X[j];

        transpose(X_batch, X_batch, batch, n);
        transpose(theta, theta_T, n, k);
        multiply(theta_T, k, n, X_batch, n, batch, Z);

        delete [] theta_T;

        for(unsigned int j=0; j<batch*k; j++)
            Z[j] = exp(Z[j]);

        transpose(Z, Z, k, batch);
        
        for(unsigned int j=0; j<batch; j++){
            float s = 0;
            for(unsigned int p=0; p<k; p++){
                s += Z[k*j+p];
            }
            for(unsigned int p=0; p<k; p++)
                Z[k*j+p] /= s;
        }
       
        for(unsigned int j=0; j<batch*k; j++)
            Z[j] = Z[j] - I[i*k*batch + j];
        
        multiply(X_batch, n, batch, Z, batch, k, gradients);

        delete [] X_batch;
        delete [] Z;

        //transpose(theta, k, n);
        
        for(unsigned int j=0; j<k*n; j++){
            theta[j] -= lr * gradients[j] / batch;
        }

        delete [] gradients;
        
    }  
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
