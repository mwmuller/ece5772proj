
#ifndef NN_FUN_H__
#define NN_FUN_H__
    
    #include <tbb/tbb.h>
    using namespace tbb; 
    
    typedef struct networkParams {
        
        // neural network parameters
        int num_layers;
        int num_inputs; 
        int num_outputs;
        int* layer_sizes; 
        double* network_biases;
        double* network_weights;
        double* network_inputs;  
        
    } NetworkParams; 
    
    
    typedef struct layerInfo {
        
        // information regarding the current layer
        int i;
        int sum_layer;
        int sum_out_layer; 
        double* a_in;
        double* a_out; 
        
    } LayerInfo; 
    
    
    class NetworkLayer {
          
        
        public: 
        
            NetworkParams params;
            LayerInfo info;
            
            NetworkLayer(NetworkParams params_in, LayerInfo info_in){
                params = params_in;
                info = info_in;
            }
            
            void operator()(const blocked_range<int> &r) const{
                
                int layerNum = info.i; 
                int num_in  = params.layer_sizes[layerNum];
                int num_out = params.layer_sizes[layerNum+1]; 
            
                for(int j = r.begin(); j != r.end(); j++){
                    
                    double b = params.network_biases[info.sum_out_layer + j];
                    
                    /* weight number */
                    for(int k = 0; k < num_in; k++){
                        double w    = params.network_weights[info.sum_layer + j*num_in + k];
                        double inp  = info.a_in[k];
                        double outp = w*inp; 
                        info.a_out[j] += outp; 
                    }
                    
                    if (layerNum != params.num_layers - 2) {
                        double result = info.a_out[j] + b;
                        /* ReLU activation function */
                        info.a_out[j] = (result > 0) ? result : 0;
                    }
                    else {
                        double result = info.a_out[j] + b;
                        /* purelin activation function */
                        info.a_out[j] = result; 
                    }
                  }
            }       
            
    };
    
    class PipelineInput {
        private:
            int num_out;
        public:
            PipelineInput(int num_out_t): num_out(num_out_t) {}
            
        int operator() (flow_control& fc) const {
            static int j = 0;
            if (j < num_out) {
                int j_out = j;
                j++;
                return j_out; 
            }
            else {
                j = 0;
                fc.stop();
            }
        }
    };
    
    class PipelineTransfer {
        public:
            NetworkParams np;
            LayerInfo inf;
            
            PipelineTransfer(NetworkParams np_i, LayerInfo inf_i): np(np_i), inf(inf_i) {}
            
            double operator() (int j) const {
                double result = 0.0;
                int layerNum = inf.i; 
                int num_in  = np.layer_sizes[layerNum];
                int num_out = np.layer_sizes[layerNum+1]; 
                double b = np.network_biases[inf.sum_out_layer + j];  
                /* weight number */
                for(int k = 0; k < num_in; k++){
                    double w    = np.network_weights[inf.sum_layer + j*num_in + k];
                    double inp  = inf.a_in[k];
                    double outp = w*inp; 
                    result += outp; 
                }
                
                /* if we ARE NOT computing the output layer */
                if (layerNum != np.num_layers - 2) {
                    double z = result + b;
                    /* ReLU activation function */
                    result = (z > 0) ? z : 0;
                    return result;
                }
                /* if we ARE computing the output layer */
                else {
                    /* purelin activation function */
                    return (result + b);
                }
            }
    };
    
    class PipelineOutput {
        public: 
            LayerInfo inf; 
            int num_out;
            
            PipelineOutput (LayerInfo inf_i, int num_out_i):
                inf(inf_i), num_out(num_out_i) {}
                
            void operator () (double result) const {
                static int cnt = 0;
                inf.a_out[cnt++] = result; 
                
                if (cnt == num_out){
                    cnt = 0; 
                }
            }
    };
    
    void RunLayerPipeline (int ntoken, int num_out, NetworkParams np, LayerInfo info) { 
        parallel_pipeline(ntoken, 
        make_filter<void, int>
(filter::serial_in_order, PipelineInput(num_out)) &
        make_filter<int, double>
(filter::parallel, PipelineTransfer(np, info)) & 
        make_filter<double, void>
(filter::serial_in_order, PipelineOutput(info, num_out)));
    }
    
    void ParallelComputeLayer(NetworkParams p, LayerInfo l, int num_out){
    parallel_for(blocked_range<int>(0, num_out), NetworkLayer(p, l));
}
    
    

#endif
