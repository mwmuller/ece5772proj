
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
#define  ENABLE_TESTING
#define  ENABLE_NETWORK
#define  STATUS_FAIL    0
#define  STATUS_SUCCESS 1

/* function signatures -- will likely be defined in a header */
/*void loadNetwork(); // "loadNetwork" fetches data from file

class NetworkLayer {

};*/



int main(){
    
    /* these parameters will be populated by the loadNetwork function */
    int num_layers  = 4;
    int num_inputs  = 2;
    int num_outputs = 1;
    int layer_sizes[] = {2, 2, 2, 1}; // input - hidden layers - output
    
    //double *network_biases   = D_CALLOC(2+2+1); // on all except input layer
    //double *network_weights  = D_CALLOC(2*2 + 2*2 + 2+1); 
    
    double network_biases[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    double network_weights[] = {0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5};
    double network_inputs[] = {1543634, -100}; 
    double *a_in, *a_out;
    
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
    
    #ifdef ENABLE_NETWORK
    
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
            
            /* output neuron number */
            for(int j = 0; j < num_out; j++){
                
                double b = network_biases[sum_out_layer + j];
                
                /* weight number */
                for(int k = 0; k < num_in; k++){
                    
                    //printf("j = %d, k = %d:\n", j, k);
                    //printf("---------------\n"); 
                    double w    = network_weights[sum_layer + k*num_out + j];
                    double inp  = a_in[k];
                    double outp = w*inp; 
                    a_out[j] += outp; 
                    //printf("w = %lf, inp = %lf, outp = %lf, a_out[%d] = %lf\n", w, inp, outp, j, a_out[j]); 
                    //printf("\n"); 
                }
                /* ReLU activation function */
                double result = a_out[j] + b;
                a_out[j] = (result > 0) ? result : 0;
            }
            sum_layer += num_in*num_out;
            sum_out_layer += num_out; 
            
            for(int l = 0; l < num_out; l++){
            printf("Layer %d: out[%d] = %lf\n", i, l, a_out[l]);
        }
        }
        
    #endif
    

}
