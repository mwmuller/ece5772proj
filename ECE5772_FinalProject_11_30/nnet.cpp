
// ECE 5772: High Performance Embedded Programming
// Authors: Luke Nuculaj and Michael Muller
// The following is a parallel_pipeline implementation of a deep 
// neural network.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <tbb/tbb.h>
#include "nn_fun_withRed.h"
#include "nnArchBias.h"
#include "nnArchWeights.h"


#define EN_SEQ
#define EN_PARFOR
#define EN_PARPIP
#define EN_PARRED

#define ENABLE_BIG_NETWORK
#define I_CALLOC(x) (int*)calloc(x, sizeof(int)); 
#define D_CALLOC(x) (double*)calloc(x, sizeof(double)); 
#define MEMCPY(dest, orig, num) for(int cnt = 0; cnt < num; cnt++) dest[cnt] = orig[cnt];

using namespace tbb; 

int main(){
    
    /*
     *  INFO: 
     *  -----
     *  ENABLE_BIG_NETWORK loads and computes the output of a pre-trained neural net
     *  with 1 input, 1 output, and two hidden layers of size 5 and 5.
     *
     *  input: -1, expected result: -1.0082 (TEST PASSED)
     *  input:  0, expected result:  0.0020 (TEST PASSED)
     */
     
    /* PARAMETER INITIALIZATION */
    int num_layers  = 4;
    int num_inputs  = 1;
    int num_outputs = 1;
    int layer_sizes[] = {1, 5, 5, 1}; // input - hidden layers - output
    
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
    double *a_in  = D_CALLOC(num_inputs);
    double *a_out = D_CALLOC(num_outputs);
    int sum_layer;
    int sum_out_layer; 
    NetworkParams params;
    LayerInfo info;
    struct timeval seq_start, seq_end, pf_start, pf_end, pp_start, pp_end;
    double seq_us = 0.0, pfor_us = 0.0, ppip_us = 0.0; 
    double seq_ans = 0.0, pfor_ans = 0.0, ppip_ans = 0.0; 

    params = 
        {
            num_layers,
            num_inputs,
            num_outputs,
            layer_sizes,
            network_biases,
            network_weights,
            network_inputs, 
            relu,
            softplus
        };  
        
    #ifdef EN_SEQ
    
        info = 
            {
                0,
                0,
                0,
                a_in,
                a_out,
            };
        
        /* FORWARD PROPAGATION */ 
        gettimeofday(&seq_start, NULL);                                
        SequentialNeuralNet(&params, &info);
        gettimeofday(&seq_end, NULL); 
        
        // timing
        seq_us += (seq_end.tv_sec - seq_start.tv_sec)*1000000 + seq_end.tv_usec - seq_start.tv_usec;
        printf("\nComputation Time (sequential): %lf\n", seq_us);
        printf("Output(s):\t%lf\n", a_out[0]); 

    #endif
    
    #ifdef EN_PARFOR

        info = 
            {
                0,
                0,
                0,
                a_in,
                a_out
            };
        /* FORWARD PROPAGATION */   
        gettimeofday(&pf_start, NULL);                      
        ParForNeuralNet(&params, &info);
        gettimeofday(&pf_end, NULL);

        a_out[0] = info.a_out[0];
        // Without the above statement, we just pass the same value along
        pfor_us += (pf_end.tv_sec - pf_start.tv_sec)*1000000 + pf_end.tv_usec - pf_start.tv_usec;
        printf("\nComputation Time (parallel_for): %lf\n", pfor_us);
        printf("Output(s):\t%lf\n", a_out[0]);

    #endif
    
    #ifdef EN_PARPIP
    
        info = 
            {
                0,
                0,
                0,
                a_in,
                a_out
            };
        
        // parallel pipelined implementation   
        gettimeofday(&pp_start, NULL);
        ParPipNeuralNet(&params, &info);
        gettimeofday(&pp_end, NULL); 
        
        a_out[0] = info.a_out[0];
        ppip_us += (pp_end.tv_sec - pp_start.tv_sec)*1000000 + pp_end.tv_usec - pp_start.tv_usec;
        printf("\nComputation Time (parallel_pipeline): %lf\n", ppip_us);
        printf("Output(s):\t%lf\n", a_out[0]);
    
    #endif

    #ifdef EN_PARRED
    
        info = 
            {
                0,
                0,
                0,
                a_in,
                a_out
            };
        // parallel reduce implementation   
        gettimeofday(&pp_start, NULL);
        ParRedNeuralNet(&params, &info);  
        gettimeofday(&pp_end, NULL); 
        a_out[0] = info.a_out[0];
        ppip_us += (pp_end.tv_sec - pp_start.tv_sec)*1000000 + pp_end.tv_usec - pp_start.tv_usec;
        printf("\nComputation Time (parallel_reduce): %lf\n", ppip_us);
        printf("Output(s):\t%lf\n", a_out[0]);
    
    #endif
    
    free(a_in);
    free(a_out); 
    
    return 0;
}
