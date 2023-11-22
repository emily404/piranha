
#pragma once

#include "../util/util.cuh"
#include "FCLayer.h"
#include "CNNLayer.h"
#include "MaxpoolLayer.h"
#include "AveragepoolLayer.h"
#include "ReLULayer.h"
#include "ResLayer.h"
#include "LNLayer.h"
#include "NeuralNetwork.h"
#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../util/functors.h"
#include <math.h>       /* log2 */
#include <sys/types.h>
#include <sys/stat.h>

extern size_t INPUT_SIZE;
extern size_t NUM_CLASSES;
extern bool WITH_NORMALIZATION;
extern bool LARGE_NETWORK;
extern size_t db_layer_max_bytes;

extern nlohmann::json piranha_config;

// input batch, labels
// get output of last layer, normalize, then subtract from labels for derivates
template<typename T, template<typename, typename...> typename Share>
NeuralNetwork<T, Share>::NeuralNetwork(NeuralNetConfig* config, int seed) : input(MINI_BATCH_SIZE * INPUT_SIZE) {

	for (int i = 0; i < config->layerConf.size(); i++) {
		if (config->layerConf[i]->type.compare("FC") == 0) {
			layers.push_back(new FCLayer<T, Share>((FCConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("CNN") == 0) {
			layers.push_back(new CNNLayer<T, Share>((CNNConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("ReLU") == 0) {
			layers.push_back(new ReLULayer<T, Share>((ReLUConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("Maxpool") == 0) {
			layers.push_back(new MaxpoolLayer<T, Share>((MaxpoolConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("LN") == 0) {
		    layers.push_back(new LNLayer<T, Share>((LNConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("Averagepool") == 0) {
		    layers.push_back(new AveragepoolLayer<T, Share>((AveragepoolConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("Res") == 0) {
            layers.push_back(new ResLayer<T, Share>((ResLayerConfig *) config->layerConf[i], i, seed+i));
        } else {
			error("Only FC, CNN, ReLU, Maxpool, Averagepool, ResLayer, and LN layer types currently supported");
        }
	}
}

template<typename T, template<typename, typename...> typename Share>
NeuralNetwork<T, Share>::~NeuralNetwork()
{
	for (auto it = layers.begin(); it != layers.end(); ++it) {
		delete (*it);
    }

	layers.clear();
}


template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::printNetwork() {
    for (auto it = layers.begin(); it != layers.end(); it++) {
        (*it)->printLayer();
    }
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::loadSnapshot(std::string path) {

    for (auto it = layers.begin(); it != layers.end(); it++) {
        (*it)->loadSnapshot(path);
    }

    //std::string input_file = path + "/input";
    //loadShareFromFile(input_file, input);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::saveSnapshot(std::string path) {

    int status = mkdir(path.c_str(), S_IRWXU | S_IRWXG); 
    /*
    //printf("directory create status: %d\n", status);
    if (errno == EEXIST && partyNum != 0) {
        return;
    }

    if (errno != 0 && errno != EEXIST) {
        printf("directory create failed with status %d\n", errno);
        exit(errno);
    }
    // TODO make sure this works on localhost
    */

    for (auto it = layers.begin(); it != layers.end(); it++) {
        (*it)->saveSnapshot(path);
    }

    printf("snapshot saved to: %s\n", path.c_str());
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::forward(std::vector<double> &data) {

    input.zero();
    input.setPublic(data);

	log_print("NN.forward");
    db_layer_max_bytes = 0;

    //printShareTensor(input, "input", 1, 1, 28, 28);
    //printShare(input, "input");

	layers[0]->forward(input);

	for (size_t i = 1; i < layers.size(); ++i) {
        db_layer_max_bytes = 0;
	
	    layers[i]->forward(*(layers[i-1]->getActivation()));
	}

    if (piranha_config["print_activations"]) {
        printShareFinite(*(layers[layers.size()-1]->getActivation()), "output activation", 10);
    }

    log_print("NN.forward_done");
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::backward(std::vector<double> &labels) {

    Share<T> labelShare(labels.size());
    labelShare.setPublic(labels);

    Share<T> deltas(labels.size());
    _backward_delta(labelShare, deltas);

    if (piranha_config["print_deltas"]) {
        printShareFinite(deltas, "input delta to bw pass", 10);
    }

    _backward_pass(deltas);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_relu_grad(Share<T> &labels, Share<T> &deltas) {

    deltas += *(layers[layers.size() - 1]->getActivation());
    deltas -= labels;
    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }

    //printShareTensor(deltas, "deltas (non-normalized)", 128, 1, 1, 10);
    //exit(1);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_relu_norm_grad(Share<T> &labels, Share<T> &deltas) {

    int nClasses = labels.size() / MINI_BATCH_SIZE;

    Share<T> mu(MINI_BATCH_SIZE);
    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            layers[layers.size() - 1]->getActivation()->getShare(share),
            mu.getShare(share), 
            false, MINI_BATCH_SIZE, nClasses
        );
    }

    //printShare(mu, "mu");

    Share<T> inversedMu(MINI_BATCH_SIZE);
    inverse(mu, inversedMu);

    //printShare(inversedMu, "inverse mu");

    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(inversedMu.getShare(share), deltas.getShare(share), nClasses);
    }

    deltas *= *(layers[layers.size() - 1]->getActivation());
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);

    //printShareTensor(deltas, "after divide", 128, 1, 1, 10);

    deltas -= labels;

    //printShareTensor(deltas, "minus labels", 128, 1, 1, 10);
    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }

    /*
    printShareTensor(deltas, "deltas (normalized)", 1, 1, 128, 10);
    exit(1);
    */
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_reveal_softmax_grad(Share<T> &labels, Share<T> &deltas) {

    Share<T> x(deltas.size());
    x += *layers[layers.size() - 1]->getActivation();

    //printShareFinite(*layers[layers.size() - 1]->getActivation(), "activations", 10);

    //printShareTensor(x, "logits", 1, 1, 1, 10);

    DeviceData<T> revealedX(x.size());
    reconstruct(x, revealedX);

    thrust::device_vector<double> floatActivations(revealedX.size());
    thrust::transform(revealedX.begin(), revealedX.end(),
            floatActivations.begin(), to_double_functor<T>());

    /*
    printf("input to softmax:\n");
    for (int i = 0; i < floatActivations.size(); i++) {
        double act = floatActivations[i];
        printf("%f ", act);
    }
    printf("\n");
    */

    int nClasses = labels.size() / MINI_BATCH_SIZE;
    thrust::device_vector<double> sums(floatActivations.size() / nClasses, 0);
    for (int i = 0; i < floatActivations.size(); i++) {
        floatActivations[i] = exp(floatActivations[i]); 
        //floatActivations[i] *= floatActivations[i]; 

        sums[i / nClasses] += floatActivations[i];
    }

    for (int i = 0; i < floatActivations.size(); i++) {
        floatActivations[i] /= sums[i / nClasses];
    }

    /*
    printf("after softmax:\n");
    for (int i = 0; i < floatActivations.size(); i++) {
        double act = floatActivations[i];
        printf("%f ", act);
    }
    printf("\n");
    */

    DeviceData<T> softmax_values(floatActivations.size());
    thrust::transform(floatActivations.begin(), floatActivations.end(), softmax_values.begin(),
        to_fixed_functor<T>());

    deltas.zero();
    deltas += softmax_values;
    deltas -= labels;
    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }

    //printShareFinite(deltas, "softmax delta", 10);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_softmax_grad(Share<T> &labels, Share<T> &deltas) {

    Share<T> x(deltas.size());
    x += *layers[layers.size() - 1]->getActivation();


    size_t n = 3;
    dividePublic(x, (T)1 << n);

    x += 1 << FLOAT_PRECISION;
    
    for (int i = 0; i < n - 1; i++) {
        x *= x;
        dividePublic(x, (T)1 << FLOAT_PRECISION);
    } 

    int nClasses = labels.size() / MINI_BATCH_SIZE;

    Share<T> sums(MINI_BATCH_SIZE);
    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            x.getShare(share),
            sums.getShare(share), 
            false, MINI_BATCH_SIZE, nClasses
        );
    }

    Share<T> inversedSums(MINI_BATCH_SIZE);

#if 1
    DeviceData<T> revealedSums(sums.size());
    reconstruct(sums, revealedSums);

    DeviceData<T> invertedRevealedSums(sums.size());

    thrust::transform(revealedSums.begin(), revealedSums.end(), invertedRevealedSums.begin(),
        inv_fixed_point_functor<T>());

    inversedSums += invertedRevealedSums;
#else
    inverse(sums, inversedSums);
#endif

    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(inversedSums.getShare(share), deltas.getShare(share), nClasses);
    }

    deltas *= x;
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);

    deltas -= labels;

    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_adhoc_softmax_grad(Share<T> &labels, Share<T> &deltas) {

    Share<T> maxVals(MINI_BATCH_SIZE);

    int numClasses = deltas.size() / MINI_BATCH_SIZE;
    int paddedSize = pow(ceil(log2(numClasses)), 2);

    Share<T> pools(MINI_BATCH_SIZE * paddedSize);
    for(int i = 0; i < Share<T>::numShares(); i++) {

        // TODO fix 4PC
        T pad_value = (T)(-10 * (1 << FLOAT_PRECISION));

        if(Share<T>::numShares() == 3) {
            switch(partyNum) {
                case 0:
                    pad_value = 0;
                    break;
                case 1:
                    if(i != 2) pad_value = 0;
                    break;
                case 2:
                    if(i != 1) pad_value = 0; 
                    break;
                case 3:
                    if(i != 0) pad_value = 0;
                    break;
            }
        }


        gpu::stride_pad(
            layers[layers.size() - 1]->getActivation()->getShare(i),
            pools.getShare(i),
            numClasses,
            paddedSize - numClasses,
            pad_value
        );
    }

    Share<uint8_t> expandedPrime(pools.size());
    maxpool(pools, maxVals, expandedPrime, paddedSize);

    //printShareFinite(maxVals, "max val", 1);

    Share<T> expandedMaxVals(deltas.size());
    for(int i = 0; i < Share<T>::numShares(); i++) {
        gpu::vectorExpand(maxVals.getShare(i), expandedMaxVals.getShare(i), numClasses);
    }

    //printShareFinite(expandedMaxVals, "expanded max val", 10);

    Share<T> diff(deltas.size());
    diff += *layers[layers.size() - 1]->getActivation();
    diff -= expandedMaxVals;
    //printShareFinite(diff, "diff", 10);

    diff += (1 << (FLOAT_PRECISION + 1));
    //printShareFinite(diff, "diff + 2", 10);

    Share<T> zeros(deltas.size());
    Share<uint8_t> b(deltas.size());
    ReLU(diff, zeros, b);
    zeros.zero();
    zeros += (T)(0.001 * (1 << FLOAT_PRECISION));

    dividePublic(diff, (T)2);

    Share<T> exponentialApprox(deltas.size());
    selectShare(zeros, diff, b, exponentialApprox);

    Share<T> sums(MINI_BATCH_SIZE);
    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            exponentialApprox.getShare(share),
            sums.getShare(share),
            false, MINI_BATCH_SIZE, numClasses
        );
    }

    DeviceData<T> revealedSums(sums.size());
    reconstruct(sums, revealedSums);

    thrust::transform(revealedSums.begin(), revealedSums.end(), revealedSums.begin(), inv_fixed_point_functor<T>());

    Share<T> inversedSums(MINI_BATCH_SIZE);
    inversedSums += revealedSums;
    //inverse(sums, inversedSums);

    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(inversedSums.getShare(share), deltas.getShare(share), numClasses);
    }

    deltas *= exponentialApprox;
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);

    deltas -= labels;

    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }

    //printShareFinite(deltas, "approx deltas", 10);
    //printf("\n");
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_backward_delta(Share<T> &labels, Share<T> &deltas) {

    //_relu_grad(labels, deltas);
    //_relu_norm_grad(labels, deltas);
    //_softmax_grad(labels, deltas);
    _reveal_softmax_grad(labels, deltas);
    //deltas.zero();
    //_adhoc_softmax_grad(labels, deltas);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_backward_pass(Share<T> &deltas) {

    // backwards pass
    layers[layers.size() - 1]->backward(
        deltas,
        *(layers[layers.size() - 2]->getActivation())
    );

	for (size_t i = layers.size() - 2; i > 0; i--) {
	    layers[i]->backward(
            *(layers[i+1]->getDelta()),
            *(layers[i-1]->getActivation())
        );
	}

    if (layers.size() > 1) {
        layers[0]->backward(
            *(layers[1]->getDelta()),
            input 
        );
    }
}

std::vector<uint32_t> sha256StringToUint32Vector(const std::string& sha256Hash) {
    std::vector<uint32_t> result;
    std::istringstream stream(sha256Hash);

    // Process the SHA-256 hash string in chunks of 8 characters (32 bits)
    while (stream.good()) {
        std::string chunk;
        stream >> std::setw(8) >> chunk; // Read 8 characters at a time

        try {
            uint32_t intValue = std::stoul(chunk, nullptr, 16);
            result.push_back(intValue);
        } catch (const std::out_of_range& ex) {
            // Handle out-of-range exception if necessary
            std::cerr << "Out of range exception caught: " << ex.what() << std::endl;
        } catch (const std::invalid_argument& ex) {
            // Handle invalid argument exception if necessary
            std::cerr << "Invalid argument exception caught: " << ex.what() << std::endl;
        }
    }

    return result;
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::jmp_verify(const int partyi, const int partyj, const int partyk, const int partyl) {
    
    size_t size = 8; // SHA-256 output 256 bit which is 8 uint32_t
    DeviceData<uint32_t> hashs(size); 
    hashs.zero();
    
    DeviceData<uint8_t> bi(1), bj(1), bk(1), bl(1), CG(2), temp(1);
    std::vector<uint8_t> host_bi(1), host_bj(1), host_bk(1), host_bl(1), host_CG(2);
    bi.zero(); bj.zero(); bk.zero(); bl.zero(); CG.zero(); temp.fill(1);
    uint8_t b;
    host_CG[0] = host_CG[1] = 4; // 4 represent no party

    if (partyNum == partyi) // sender in sending phase
    {
        // 1. receive bit from party k
        bi.receive(partyk);
        bi.join();
        // 2. exchange b with party j, party l
        // bk[0] = bi[0];
        bk.zero();
        bk += bi;
        comm_profiler.start();
        bk.transmit(partyj);
        bi.transmit(partyl);
        bj.receive(partyj);
        bl.receive(partyl);
        bk.join();
        bi.join();
        bj.join();
        bl.join();
        comm_profiler.accumulate("comm-time");

        // 3. compute a majority bit
        copyToHost(bi, host_bi, false);
        copyToHost(bj, host_bj, false);
        copyToHost(bl, host_bl, false);
        b = ((host_bi[0] + host_bj[0] + host_bl[0]) >=2 );
        // 4. if b=1, send hashs to party l
        if (b == 1)
        {
            std::string sendHash = getSendHash(partyi, partyj);
            std::vector<uint32_t> hashs_host = sha256StringToUint32Vector(sendHash);
            thrust::copy(hashs_host.begin(), hashs_host.end(), hashs.begin());

            comm_profiler.start();
            hashs.transmit(partyl);
            hashs.join();
            
            // 5. receive CG from party l
            CG.receive(partyl);
            CG.join();
            comm_profiler.accumulate("comm-time");
            printDeviceData(CG, "CG", false);
        }
    }
    else if (partyNum == partyj) // sender in verifying phase
    {
        comm_profiler.start();
        // 1. send hj to party k
        std::string verifyHash = getVerifyHash(partyi, partyj);
        std::vector<uint32_t> hashs_j_host = sha256StringToUint32Vector(verifyHash);
        thrust::copy(hashs_j_host.begin(), hashs_j_host.end(), hashs.begin());
        
        hashs.transmit(partyk);
        hashs.join();
        
        // 2. receive bit from party k
        bj.receive(partyk);
        bj.join();
        // 3. exchange b with party i, party l
        // bk[0] = bj[0];
        bk.zero();
        bk += bj;
        bk.transmit(partyi);
        bj.transmit(partyl);
        bi.receive(partyi);
        bl.receive(partyl);
        bk.join();
        bj.join();
        bi.join();
        bl.join();
        comm_profiler.accumulate("comm-time");
        // 4. compute a majority bit
        copyToHost(bi, host_bi, false);
        copyToHost(bj, host_bj, false);
        copyToHost(bl, host_bl, false);
        b = ((host_bi[0] + host_bj[0] + host_bl[0]) >=2 );
        // 5. if b=1, send hashs to party l
        if (b == 1)
        {
            comm_profiler.start();
            hashs.transmit(partyl);
            hashs.join();

            // 5. receive CG from party l
            CG.receive(partyl);
            CG.join();
            comm_profiler.accumulate("comm-time");
            printDeviceData(CG, "CG", false);
        }
    }
    else if (partyNum == partyk) // receiver
    {
        DeviceData<uint32_t> hashs_j(size);
        std::vector<uint32_t> host_hashs_j(size);
        std::string receiveHash = getReceiveHash(partyi, partyj);
        std::vector<uint32_t> host_hashs = sha256StringToUint32Vector(receiveHash);

        // 1. receive hj from party j
        comm_profiler.start();
        hashs_j.receive(partyj);
        hashs_j.join();
        comm_profiler.accumulate("comm-time");
        copyToHost(hashs_j, host_hashs_j, false);
        
        // 2. check the consistency of hj and hashs
        for(auto i = 0; i < size; i++)
        {
            if (host_hashs_j[i] != host_hashs[i])
            {
                bi.zero(); bj.zero(); bk.zero(); bl.zero();
                bi += temp; bj += temp; bk += temp; bl += temp;
                break;
            }
        }
        // 3. send bit to party party i, party j, party l
        comm_profiler.start();
        bi.transmit(partyi);
        bj.transmit(partyj);
        bl.transmit(partyl);
        bi.join();
        bj.join();
        bl.join();
        comm_profiler.accumulate("comm-time");

        // 4. if b=1, send hashs to party l
        copyToHost(bi, host_bi, false);
        if (host_bi[0] == 1)
        {
            // concat hashs and hashs_j in a new vector
            DeviceData<uint32_t> hashs_l(2*size);
            thrust::copy(host_hashs.begin(), host_hashs.end(), hashs_l.begin());
            thrust::copy(host_hashs_j.begin(), host_hashs_j.end(), hashs_l.begin() + size);

            comm_profiler.start();
            hashs_l.transmit(partyl);
            hashs_l.join();
    
            // 5. receive CG from party l
            CG.receive(partyl);
            CG.join();
            comm_profiler.accumulate("comm-time");
            printDeviceData(CG, "CG", false);
        }
    }
    else if (partyNum == partyl) // candidates of TTP
    {
        // 1. receive bit from party k
        comm_profiler.start();
        bl.receive(partyk);
        bl.join();
        // 2. exchange b with party i, party j
        // bk[0] = bl[0];
        bk.zero();
        bk += bl;
        bk.transmit(partyi);
        bl.transmit(partyj);
        bi.receive(partyi);
        bj.receive(partyj);
        bk.join();
        bl.join();
        bi.join();
        bj.join();
        comm_profiler.accumulate("comm-time");
        // 3. compute a majority bit
        copyToHost(bi, host_bi, false);
        copyToHost(bj, host_bj, false);
        copyToHost(bl, host_bl, false);
        b = ((host_bi[0] + host_bj[0] + host_bl[0]) >=2 );
        // 4. if b=1, receive hashs from party i, party j, party k
        if (b == 1)
        {
            DeviceData<uint32_t> hashs_i(size), hashs_j(size), hashs_k(2*size);
            hashs_i.zero(); hashs_j.zero(); hashs_k.zero();
            std::vector<uint32_t> host_hashs_i(size);
            std::vector<uint32_t> host_hashs_j(size);
            std::vector<uint32_t> host_hashs_k(2*size);
            comm_profiler.start();
            hashs_i.receive(partyi);
            hashs_j.receive(partyj);
            hashs_k.receive(partyk);
            hashs_i.join();
            hashs_j.join();
            hashs_k.join();
            comm_profiler.accumulate("comm-time");
            copyToHost(hashs_i, host_hashs_i, false);        
            copyToHost(hashs_j, host_hashs_j, false);        
            copyToHost(hashs_k, host_hashs_k, false);        
            
            // check the consistency of hashs_i, hashs_j, hashs_ik, hashs_jk
            uint8_t check_bit0, check_bit1, check_bit2, check_bit3;
            check_bit0 = check_bit1 = check_bit2 = check_bit3 = 1;
            // 4.1 check the consistency of hashs_ik and hashs_jk 
            for(auto i = 0; i < size; i++)
            {
                if (host_hashs_k[i] != host_hashs_k[i+size])
                {
                    check_bit0 = 0;
                    break;
                }
            }
            if (check_bit0) // if equal, output CG={partyk}
            {
                host_CG[0] = partyk;
            }
            else // else
            {
                // 4.2 check the consistency of hashs_i and hashs_ik
                // 4.3 check the consistency of hashs_j and hashs_jk
                for(auto i = 0; i < size; i++)
                {
                    if (host_hashs_i[i] != host_hashs_k[i])
                    {
                        check_bit1 = 0;
                    }
                    if (host_hashs_j[i] != host_hashs_k[i+size])
                    {
                        check_bit2 = 0;
                    }
                    if (check_bit1 == 0 && check_bit2 == 0)
                    {
                        break;
                    }
                }
                if (check_bit1 == 0 && check_bit2 == 0) // if both not equal, output CG={partyk}
                {
                    host_CG[0] = partyk;
                }
                else if (check_bit1 == 1 && check_bit2 == 1) // if both equal, output CG={partyi,partyj}
                {
                    host_CG[0] = partyi;
                    host_CG[1] = partyj;
                }
                else // otherwise, check the consistency of hashs_i and hashs_j
                {
                    for(auto i = 0; i < size; i++)
                    {
                        if (host_hashs_i[i] != host_hashs_j[i])
                        {
                            check_bit3 = 0;
                            break;
                        }
                    }
                    host_CG[0] = check_bit2 == 1 ? partyi : partyj;
                    host_CG[1] = check_bit3 == 1 ? partyk : 4;
                }
            }

            // copy several CG to send in parallel
            DeviceData<uint8_t> CG_i(2), CG_j(2), CG_k(2);
            thrust::copy(host_CG.begin(), host_CG.end(), CG_i.begin());
            thrust::copy(host_CG.begin(), host_CG.end(), CG_j.begin());
            thrust::copy(host_CG.begin(), host_CG.end(), CG_k.begin());
            printDeviceData(CG_i, "CG_i", false);

            // 5. send CG to other parties
            comm_profiler.start();
            CG_i.transmit(partyi);
            CG_j.transmit(partyj);
            CG_k.transmit(partyk);
            CG_i.join();
            CG_j.join();
            CG_k.join();
            comm_profiler.accumulate("comm-time");
        }
    }
    else
    {
        assert(false && "jmp_verify called on incorrect party");
    }
}

/*
template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::printLoss(std::vector<double> &labels, bool cross_entropy) {
    
	layers[i]->forward(*(layers[i-1]->getActivation()));
    DeviceData<T> reconstructedOutput(
    reconstruct(outputData, reconstructedOutput);
    std::vector<double> host_output(reconstructedOutput.size());
    copyToHost(reconstructedOutput, host_output, true);

    std::vector<double> host_expected(expectedOutput.size());
    copyToHost(expectedOutput, host_expected, true);

    double cumulative_error = 0.0;
    for(int i = 0; i < host_output.size(); i++) {
        if (cross_entropy) {
            if (host_expected[i] == 1) {
                cumulative_error -= log2 (host_output[i]);
            }
        } else {
            cumulative_error += fabs(host_output[i] - host_expected[i]);
        }
    }

    if (cross_entropy) {
    	std::cout << "cross entropy loss from expected FW pass results: " << cumulative_error << std::endl;
    } else {
    	std::cout << "cumulative error from expected FW pass results: " << cumulative_error << std::endl;
    }
    
    std::cout << "expected (first 10): ";
    for (int i = 0; i < 10; i++) std::cout << host_expected[i] << " ";
    std::cout << std::endl;

    std::cout << "actual (first 10): ";
    for (int i = 0; i < 10; i++) std::cout << host_output[i] << " ";
    std::cout << std::endl;
}
*/

template class NeuralNetwork<uint32_t, RSS>;
template class NeuralNetwork<uint64_t, RSS>;

template class NeuralNetwork<uint32_t, TPC>;
template class NeuralNetwork<uint64_t, TPC>;

template class NeuralNetwork<uint32_t, FPC>;
template class NeuralNetwork<uint64_t, FPC>;

template class NeuralNetwork<uint32_t, OPC>;
template class NeuralNetwork<uint64_t, OPC>;

