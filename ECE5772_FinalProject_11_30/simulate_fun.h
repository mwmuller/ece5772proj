
#ifndef NN_FUN_H__
#define NN_FUN_H__
    
    #include <tbb/tbb.h>
    #include <math.h>
    #define I_CALLOC(x) (int*)calloc(x, sizeof(int)); 
    #define D_CALLOC(x) (double*)calloc(x, sizeof(double)); 
    #define MEMCPY(dest, orig, num) for(int cnt = 0; cnt < num; cnt++) dest[cnt] = orig[cnt];
    
    using namespace tbb; 
    
    /* enumerated value "LAYER_ACTIVATION" for activation function */ 
    typedef enum {
        relu = 0,
        leakyrelu,
        softplus,
        purelin
    } LAYER_ACTIVATION;
    
    /* activation function definitions */ 
    double ReLU (double inp){
        return (inp > 0) ? inp : 0; 
    }
    
    double LeakyReLU (double inp) {
        double scale = 0.01;
        double result = (inp > 0) ? inp: scale*inp; 
        return result; 
    }
    
    double SoftPlus (double inp) {
        return log(1 + exp(inp)); 
    }
    
    double PureLin (double inp){
        return inp;
    }

    void printOutput(double *a_out, int num_out)
    {
        int i;
        for(int i = 0; i < num_out; i++)
        {
            printf("a_out[%d]: %lf\n", i, a_out[i]);
        }
    }
    
    double (*actfcn_arr[])(double) = {ReLU, LeakyReLU, SoftPlus, PureLin};
    
    
    typedef struct networkParams {
        
        // neural network parameters
        int num_layers;
        int num_inputs; 
        int num_outputs;
        int* layer_sizes; 
        const double* network_biases;
        const double* network_weights;
        double* network_inputs;  
        LAYER_ACTIVATION hiddenfcn;
        LAYER_ACTIVATION outfcn;
    } NetworkParams; 
    
    
    typedef struct layerInfo {
        
        // information regarding the current layer
        int i;
        int sum_layer;
        int sum_out_layer; 
        double* a_in;
        double* a_out;
        double* a_prev;
    } LayerInfo; 
    
    class NetworkLayerReduce
    {
        private:
            LayerInfo* info;
            NetworkParams* params;
            int my_weight_idx_base;
        public:
            double my_aout;
            void operator()(const blocked_range<int> &r) {
                
                int layerNum = info->i; 
                int k;
                for(k = r.begin(); k != r.end(); k++)
                {
                    double w    = params->network_weights[my_weight_idx_base + k];      
                    double inp  = info->a_in[k]; //get from info->a_in[k]
                    //printf("inside reduce a_out %lf\n", a_out);
                    my_aout += w*inp;
                }
            }  

        NetworkLayerReduce(NetworkLayerReduce &x, tbb::split) : 
        params(x.params),
        info(x.info),
        my_aout(0), // CANNOT BE x.my_aout
        my_weight_idx_base(x.my_weight_idx_base)
        {
        }
        void join (const NetworkLayerReduce &y)
        {
            my_aout += y.my_aout; // increment the value
        }
        
        NetworkLayerReduce(NetworkParams* params_in, LayerInfo* info_in, int weight_idx) : 
        params(params_in), 
        info(info_in),
        my_aout(0),
        my_weight_idx_base(weight_idx)
        {}
    };

    class NetworkLayer {
        
        public: 
        
            NetworkParams* params;
            LayerInfo* info;
            
            NetworkLayer(NetworkParams* params_in, LayerInfo* info_in){
                params = params_in;
                info = info_in;
            }
            
            void operator()(const blocked_range<int> &r) const{
                
                int layerNum = info->i; 
                int num_in  = params->layer_sizes[layerNum];
                int num_out = params->layer_sizes[layerNum+1]; 
            
                for(int j = r.begin(); j != r.end(); j++){
                    
                    double b = params->network_biases[info->sum_out_layer + j];
                    
                    // weight number  
                    for(int k = 0; k < num_in; k++){
                        double w    = params->network_weights[info->sum_layer + j*num_in + k];
                        double inp  = info->a_in[k];
                        double outp = w*inp; 
                        //printf("BaseInpu for weights %d\n", info->sum_layer + j*num_in + k);
                        info->a_out[j] += outp; 
                    }
                    
                    if (layerNum != params->num_layers - 2) {
                        double result = info->a_out[j] + b;
                        info->a_out[j] = (*actfcn_arr[params->hiddenfcn])(result); 
                    }
                    else {
                        double result = info->a_out[j] + b;
                        info->a_out[j] = (*actfcn_arr[params->outfcn])(result); 
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
            int j_out = 0;
            static int j = 0;
            //printf("j = %d | j_out %d\n", j, j_out);
            if (j < num_out) {
                j_out = j;
                j++;
                return j_out; 
            }
            else {
                j = 0;
                fc.stop();
            }
            return j_out;
        }
    };
    
    class PipelineTransfer {
        public:
            NetworkParams* np;
            LayerInfo* inf;
            
            PipelineTransfer(NetworkParams* np_i, LayerInfo* inf_i): np(np_i), inf(inf_i) {}
            
            double operator() (int j) const {
                double result = 0.0;
                int layerNum = inf->i; 
                int num_in  = np->layer_sizes[layerNum];
                int num_out = np->layer_sizes[layerNum+1]; 
                double b = np->network_biases[inf->sum_out_layer + j];
                /* weight number */
                //printf("Params: layerNum %d | num_in %d | num_out %d\n", layerNum, num_in, num_out);
                for(int k = 0; k < num_in; k++){
                    double w    = np->network_weights[inf->sum_layer + j*num_in + k];
                    //printf("Weight %lf\n", w);
                    double inp  = inf->a_in[k];
                    double outp = w*inp; 
                    result += outp; 
                }
                /* if we ARE NOT computing the output layer */
                if (layerNum != np->num_layers - 2) {
                    double z = result + b;
                    //printf("Z relu: %lf\n", z);
                    return (*actfcn_arr[np->hiddenfcn])(z); 

                }
                /* if we ARE computing the output layer */
                else {
                    /* purelin activation function */
                    double z = result + b; 
                    //printf("softLayer %lf\n", result);
                    return (*actfcn_arr[np->outfcn])(z);
                }
            }
    };
    
    class PipelineOutput {
        public: 
            LayerInfo* inf; 
            int num_out;
            
            PipelineOutput (LayerInfo* inf_i, int num_out_i):
                inf(inf_i), num_out(num_out_i) {}
                
            void operator () (double result) const {
                static int cnt = 0;
                inf->a_out[cnt++] = result; 
                if (cnt == num_out){
                    cnt = 0; 
                }
            }
    };
    
    void RunLayerPipeline (int ntoken, int num_out, NetworkParams* np, LayerInfo* info) { 
        parallel_pipeline(ntoken, 
        make_filter<void, int>
(filter::serial_in_order, PipelineInput(num_out)) &
        make_filter<int, double>
(filter::parallel, PipelineTransfer(np, info)) & 
        make_filter<double, void>
(filter::serial_in_order, PipelineOutput(info, num_out)));
    }
    
    void ParallelComputeLayer(NetworkParams* p, LayerInfo* l, int num_out){
        parallel_for(blocked_range<int>(0, num_out), NetworkLayer(p, l));
    }
    
    void ReduceComputeLayer(NetworkParams* p, LayerInfo* l, int num_out, int num_in){ // is called on each layer
        parallel_for(blocked_range<int>(0, num_out), [&] (const blocked_range<int> u){
            int j = 0;            
            for (j = u.begin(); j != u.end(); j++) // iterates through each node
            {
                // get bias to be added after MAC function 
                double b = p->network_biases[l->sum_out_layer + j];
                
                // find weight base for a node. 
                // 'split' function in 'reduce' will increment from the base idx
                // to find the correct weigth idx
                int weight_idx_base = l->sum_layer + j*num_in;
                NetworkLayerReduce nnReduce(p, l, weight_idx_base);
                // performs reduction on each weight for a given node 'j'
                parallel_reduce(blocked_range<int>(0, num_in, num_in), nnReduce);
                // override reduction output for relu functions?
                if (l->i != p->num_layers - 2) {
                        double result = nnReduce.my_aout + b;
                        l->a_out[j] = (*actfcn_arr[p->hiddenfcn])(result); 
                    }
                else {
                    double result = nnReduce.my_aout + b;
                    l->a_out[j] = (*actfcn_arr[p->outfcn])(result); 
                }
            }
        });
    }

    void SequentialNeuralNet(NetworkParams* params, LayerInfo* info) {
        
        int sum_layer = 0;
        int sum_out_layer = 0; 
        double *a_in_t, *a_out_t;
        for(int i = 0; i < params->num_layers - 1; i++){
            int num_in  = params->layer_sizes[i];
            int num_out = params->layer_sizes[i+1];

            if (i == 0){
                sum_layer = 0;
                sum_out_layer = 0;
                a_in_t  = D_CALLOC(num_in);
                MEMCPY(a_in_t, params->network_inputs, num_in); 
                a_out_t = D_CALLOC(num_out); 
            }
            else{
                free(a_in_t);
                a_in_t = D_CALLOC(num_in);
                MEMCPY(a_in_t, a_out_t, num_in);
                free(a_out_t);
                a_out_t = D_CALLOC(num_out); 
            }
            
            /* updating layer information */
            for(int j = 0; j < num_out; j++){
                double b = params->network_biases[sum_out_layer + j];
                
                for(int k = 0; k < num_in; k++){
                    
                    //printf("j = %d, k = %d:\n", j, k);
                    //printf("---------------\n"); 
                    double w    = params->network_weights[sum_layer + j*num_in + k];
                    double inp  = a_in_t[k];
                    double outp = w*inp; 
                    a_out_t[j] += outp; 
                    //printf("w = %lf, inp = %lf, w*inp = %lf\n", w, inp, outp); 
                    //printf("\n"); 
                }
                double result = 0.0;
                
                if (i != params->num_layers - 2) {
                     result = a_out_t[j] + b;
                     /* ReLU activation function */
                     a_out_t[j] = (*actfcn_arr[params->hiddenfcn])(result); 
                }
                else {
                     result = a_out_t[j] + b;
                     result = (*actfcn_arr[params->outfcn])(result); 
                     a_out_t[j] = result; 
                }
                
                if (i == -1){
                printf("\n-----------------\n");
                printf("Layer %d, a_out[%d] = %lf\n", i, j, a_out_t[j]);
                }
                //printf("b = %lf\n", b);
                //printf("Result = %lf\n", result);
                //printf("\n-----------------\n");
            }

            sum_layer += num_in*num_out;
            sum_out_layer += num_out; 
        }
        
        MEMCPY(info->a_out, a_out_t, params->num_outputs);
        free(a_in_t);
        free(a_out_t);
        
    }
    
    void ParForNeuralNet(NetworkParams* params, LayerInfo* info){
        
        int sum_layer = 0;
        int sum_out_layer = 0; 
        double *a_in_t, *a_out_t;
        
        for(int i = 0; i < params->num_layers - 1; i++){
            
            int num_in  = params->layer_sizes[i];
            int num_out = params->layer_sizes[i+1];
            
            if (i == 0){
                sum_layer = 0;
                sum_out_layer = 0;
                a_in_t  = D_CALLOC(num_in);
                MEMCPY(a_in_t, params->network_inputs, num_in); 
                a_out_t = D_CALLOC(num_out); 
            }
            else{
                free(a_in_t);
                a_in_t = D_CALLOC(num_in);
                MEMCPY(a_in_t, a_out_t, num_in);
                free(a_out_t);
                a_out_t = D_CALLOC(num_out); 
            }
            
            info->i = i;
            info->sum_layer = sum_layer;
            info->sum_out_layer = sum_out_layer; 
            info->a_in = a_in_t;
            info->a_out = a_out_t;
            
            ParallelComputeLayer(params, info, num_out);
            sum_layer += num_in*num_out;
            sum_out_layer += num_out;
            
        }
        //printf("awdawdasd %lf\n", info->a_out[0]);
       // printf("lets guess %lf\n", a_out_t[0]);
        MEMCPY(info->a_out, a_out_t, params->num_outputs);
        //free(a_in_t); // this clears info->aout since memcpy doesn't redefine the address
        //free(a_out_t);
       // printf("awdawdasd %lf\n", info->a_out[0]);
        //printf("lets guess %lf\n", a_out_t[0]);
         
    }
    
    void ParPipNeuralNet(NetworkParams* params, LayerInfo* info){
        
        int sum_layer = 0;
        int sum_out_layer = 0; 
        double *a_in_t, *a_out_t;
        
        for(int i = 0; i < params->num_layers - 1; i++){
            
            int num_in  = params->layer_sizes[i];
            int num_out = params->layer_sizes[i+1];
            
            if (i == 0){
                sum_layer = 0;
                sum_out_layer = 0;
                a_in_t  = D_CALLOC(num_in);
                MEMCPY(a_in_t, params->network_inputs, num_in); 
                a_out_t = D_CALLOC(num_out); 
            }
            else{
                free(a_in_t);
                a_in_t = D_CALLOC(num_in);
                MEMCPY(a_in_t, a_out_t, num_in);
                free(a_out_t);
                a_out_t = D_CALLOC(num_out); 
            }
            
            info->i = i;
            info->sum_layer = sum_layer;
            info->sum_out_layer = sum_out_layer; 
            info->a_in = a_in_t;
            info->a_out = a_out_t;
            
            int ntoken = 20;
            RunLayerPipeline(ntoken, num_out, params, info); 
            sum_layer += num_in*num_out;
            sum_out_layer += num_out;
        }
        MEMCPY(info->a_out, a_out_t, params->num_outputs);
        //free(a_in_t);
        //free(a_out_t);
         
    }
     
    void ParRedNeuralNet(NetworkParams* params, LayerInfo* info){
        
        int sum_layer = 0;
        int sum_out_layer = 0; 
        double *a_in_t, *a_out_t;
        
        for(int i = 0; i < params->num_layers - 1; i++){
            
            int num_in  = params->layer_sizes[i];
            int num_out = params->layer_sizes[i+1];
            if (i == 0){
                sum_layer = 0;
                sum_out_layer = 0;
                a_in_t  = D_CALLOC(num_in);
                MEMCPY(a_in_t, params->network_inputs, num_in); 
                a_out_t = D_CALLOC(num_out); 
            }
            else{
                free(a_in_t);
                a_in_t = D_CALLOC(num_in);
                MEMCPY(a_in_t, a_out_t, num_in);
                free(a_out_t);
                a_out_t = D_CALLOC(num_out); 
            }
            
            info->i = i;
            info->sum_layer = sum_layer;
            info->sum_out_layer = sum_out_layer; 
            info->a_in = a_in_t;
            info->a_out = a_out_t;

            ReduceComputeLayer(params, info, num_out, num_in);
            sum_layer += num_in*num_out;
            sum_out_layer += num_out;
            
        }
        MEMCPY(info->a_out, a_out_t, params->num_outputs);
        //free(a_in_t);
        //free(a_out_t);
         
    }
    
    void updateBatteryPack(double *state_of_charge, double *current, int num_cells){
        
        for(int i=0; i<num_cells; i++){
            state_of_charge[i] = state_of_charge[i] - current[i]/(3600.0*0.027*4.1);
        }
    }

#endif
