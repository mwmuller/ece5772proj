
// ECE 5772: High Performance Embedded Programming
// Authors: Luke Nuculaj and Michael Muller
// The following is a sequential implementation of a deep 
// neural network, which functions as the control group
// for the parallel implementation.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#define I_CALLOC(x) (int*)calloc(x, sizeof(int)); 
#define D_CALLOC(x) (double*)calloc(x, sizeof(double)); 
#define MEMCPY(dest, orig, num) for(int cnt = 0; cnt < num; cnt++) dest[cnt] = orig[cnt];
//#define  ENABLE_TESTING
//#define  ENABLE_SMALL_NETWORK
#define  ENABLE_BIG_NETWORK
#define  STATUS_FAIL    0
#define  STATUS_SUCCESS 1

/* function signatures -- will likely be defined in a header */
/*void loadNetwork(); // "loadNetwork" fetches data from file

class NetworkLayer {

};*/



int main(){
    
    
    // TESTS
    #ifdef ENABLE_TESTING
    
        // TEST 1: MEMCPY
        int status_flag = STATUS_SUCCESS; 
        double *x1 = D_CALLOC(100);
        double *x2 = D_CALLOC(100);
        for(int i = 0; i < 100; i++){
            x2[i] = i;
        }
        MEMCPY(x1, x2, 100);
        
        for(int i = 0; i < 100; i++){
            if(x1[i] != x2[i]){
                status_flag = STATUS_FAIL; 
                break;
            }
        }
        
        if (status_flag == STATUS_SUCCESS) {
            printf("TEST 1: MEMCPY SUCCESSFUL!\n");
        }
        else{
            printf("TEST 1: MEMCPY UNSUCCESSFUL!\n");
        }
        
    #endif 
    
    /*
     *  INFO: 
     *  -----
     *  ENABLE_SMALL_NETWORK loads and computes the output of a pre-trained neural net
     *  with 1 input, 1 output, and two hidden layers of size 3 and 2.
     *
     *  input: -1, expected result: -1.0527 (TEST PASSED)
     */
    #ifdef ENABLE_SMALL_NETWORK
    
        /* these parameters will be populated by the loadNetwork function */
        int num_layers  = 4;
        int num_inputs  = 1;
        int num_outputs = 1;
        int layer_sizes[] = {1, 3, 2, 1}; // input - hidden layers - output
        
        
        double network_biases[] = 
                {
                    -0.7505, 0.5159, 0.1329,
                    0.0154, 0.2186,
                    0.2963
                };
        double network_weights[] = 
                {   
                    // input layer weights
                    0.2367, -0.3791, 2.1076,
                    // hidden layer weights
                    0.6663, 0.9208, -1.8223,
                    -0.2034, 0.0909, 0.8266,
                   // output layer weights
                   -1.7417, 0.3775         
                };
        double network_inputs[] = {-1}; 
        double *a_in, *a_out;
        
    #endif
    
    /*
     *  INFO: 
     *  -----
     *  ENABLE_SMALL_NETWORK loads and computes the output of a pre-trained neural net
     *  with 1 input, 1 output, and two hidden layers of size 5 and 5.
     *
     *  input: -1, expected result: -1.0082 (TEST PASSED)
     *  input:  0, expected result:  0.0020 (TEST PASSED)
     */
    #ifdef ENABLE_BIG_NETWORK
        /* these parameters will be populated by the loadNetwork function */
        int num_layers  = 4;
        int num_inputs  = 1;
        int num_outputs = 1;
        int layer_sizes[] = {1, 5, 5, 1}; // input - hidden layers - output
        
        //double *network_biases   = D_CALLOC(2+2+1); // on all except input layer
        //double *network_weights  = D_CALLOC(2*2 + 2*2 + 2+1); 
        
        double network_biases[] = 
                {
                    0.2726, -0.8559, 0.1930, 0.0695, 0.7407,
                    -0.5903, -0.1090, -0.3654, 0.2525, -0.3625,
                    0.5746
                };
        double network_weights[] = 
                {   
                    // input layer weights
                    0.7700,
                    0.7451, 
                    0.1640,
                    2.3107,
                    -0.3672,
                    
                    // hidden layer weights
                    0.6733, 0.9759, 0.5582, -0.2056, 0.7284,
                    0.2449, 0.7283, 0.0829, 1.0228, 0.3529,
                    0.3444, -0.2222, 0.7856, -0.2770, -0.4306,
                    -0.4834, -0.0905, -0.1762, -2.2559, 0.3513,
                    1.0349, -0.5066, -0.1862, 0.9359, 0.0987,
                    
                   // output layer weights
                   -0.0913, -0.4448, 0.1683, -2.2543, 0.5929         
                };
        double network_inputs[] = {-1}; 
        double *a_in, *a_out;
    
    #endif
    
    int sum_layer = 0;
        int sum_out_layer = 0; 
        
        /* layer number */
        for(int i = 0; i < num_layers - 1; i++){
            
            int num_in  = layer_sizes[i];
            int num_out = layer_sizes[i+1]; 
            
            /* we use two arrays to hold the inputs and outputs
            to a layer at any given time. This performs the 
            memory copying and allocating when a new layer is 
            arrived at*/
            if (i == 0){
                a_in  = D_CALLOC(num_in);
                MEMCPY(a_in, network_inputs, num_in); 
                a_out = D_CALLOC(num_out); 
            }
            else{
                free(a_in);
                a_in = D_CALLOC(num_in);
                MEMCPY(a_in, a_out, num_in);
                free(a_out);
                a_out = D_CALLOC(num_out); 
            }
            
            /* output neuron number; this loop can be made parallel */
            for(int j = 0; j < num_out; j++){
                
                double b = network_biases[sum_out_layer + j];
                
                /* weight number */
                for(int k = 0; k < num_in; k++){
                    
                    //printf("j = %d, k = %d:\n", j, k);
                    //printf("---------------\n"); 
                    double w    = network_weights[sum_layer + j*num_in + k];
                    double inp  = a_in[k];
                    double outp = w*inp; 
                    a_out[j] += outp; 
                    //printf("w = %lf, inp = %lf, outp = %lf, a_out[%d] = %lf\n", w, inp, outp, j, a_out[j]); 
                    //printf("\n"); 
                }
                
                if (i != num_layers - 2) {
                    double result = a_out[j] + b;
                    /* ReLU activation function */
                    a_out[j] = (result > 0) ? result : 0;
                }
                else {
                    double result = a_out[j] + b;
                    /* purelin activation function */
                    //printf("\nPurelin here!\n");
                    a_out[j] = result; 
                }
            }
            sum_layer += num_in*num_out;
            sum_out_layer += num_out; 
            
            for(int l = 0; l < num_out; l++){
            printf("Layer %d: out[%d] = %lf\n", i, l, a_out[l]);
        }
        }
}
