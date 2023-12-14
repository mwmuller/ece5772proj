
// ECE 5772: High Performance Embedded Programming
// Authors: Luke Nuculaj and Michael Muller
// The following is a parallel_pipeline implementation of a deep 
// neural network.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <tbb/tbb.h>
#include "nn_fun.h"
#include "nnArchBias_5.h"
#include "nnArchWeights_5.h"


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
    int num_layers  = 6;
    int num_inputs  = 5;
    int num_outputs = 5;
    int layer_sizes[] = {5, 64, 64, 64, 64, 5}; // input - hidden layers - output
    
    /*
     double network_biases[] = 
            {
               0.6174, -0.9363, -0.4462, -0.9077, -0.8057,
               -1.0392, -1.2578, -0.3869, -0.0865, 1.3236,
               2.2620
            };
     double network_weights[] = 
            {   
               2.4611, 0.4863, -0.2155, 0.3110, -0.6576,
               
               1.9185, -0.1225, -0.0205, -0.4479, -0.0033,
               -1.2188, -0.2369, -0.1088, 0.3594, 0.9195,
               -0.3734, 0.5310, 0.2926, 0.3102, -0.3192,
               2.0691, 0.5904, 0.4187, -0.6748, 0.1705,
               -1.8475, -0.6263, 0.5094, -0.7620, -0.5524,
               
               2.0718, -2.5593, -0.7200, -1.7937, -2.3317
            };
    */
           
    //double network_inputs[] = {0.8147, 0.9035, 0.1292, 0.9111, 0.6346}; 
    double network_inputs[] = {0.7534, 0.9806, -1, 1, 0.2927}; 

    
    double *a_in  = D_CALLOC(num_inputs);
    double *a_out = D_CALLOC(num_outputs);
    int sum_layer;
    int sum_out_layer; 
    NetworkParams params;
    LayerInfo info;
    struct timeval seq_start, seq_end, pf_start, pf_end, pp_start, pp_end, pr_start, pr_end;
    double seq_us = 0.0, pfor_us = 0.0, ppip_us = 0.0, pred_us = 0.0; 
    double seq_ans = 0.0, pfor_ans = 0.0, ppip_ans = 0.0, pred_ans = 0.0; // Maybe not needed

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
            purelin
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
        seq_us = (seq_end.tv_sec - seq_start.tv_sec)*1000000 + seq_end.tv_usec - seq_start.tv_usec;
        printf("\nComputation Time (sequential): %lf\n", seq_us);
        printf("Output(s)\n"); 

        printOutput(info.a_out, num_outputs);

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
        pfor_us = (pf_end.tv_sec - pf_start.tv_sec)*1000000 + pf_end.tv_usec - pf_start.tv_usec;
        printf("\nComputation Time (parallel_for): %lf\n", pfor_us);
        printf("Output(s)\n");

        printOutput(info.a_out, num_outputs);

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
        ppip_us = (pp_end.tv_sec - pp_start.tv_sec)*1000000 + pp_end.tv_usec - pp_start.tv_usec;
        printf("\nComputation Time (parallel_pipeline): %lf\n", ppip_us);
        printf("Output(s)\n");

        printOutput(info.a_out, num_outputs);
    
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
        gettimeofday(&pr_start, NULL);
        ParRedNeuralNet(&params, &info);  
        gettimeofday(&pr_end, NULL); 
        a_out[0] = info.a_out[0];
        pred_us = (pr_end.tv_sec - pr_start.tv_sec)*1000000 + pr_end.tv_usec - pr_start.tv_usec;
        printf("\nComputation Time (parallel_reduce): %lf\n", pred_us);
        printf("Output(s)\n");

        printOutput(info.a_out, num_outputs);
    
    #endif
    
    free(a_in);
    free(a_out); 
    
    return 0;
}
