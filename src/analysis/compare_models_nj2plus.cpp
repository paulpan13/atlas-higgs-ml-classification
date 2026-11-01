/**
 * @file compare_models_nj2plus.cpp
 * @brief Trains and compares the final DNN and BDT classifiers
 *        for the n_j >= 2 category.
 *
 * Retrains the two classifiers using the hyperparameters selected during
 * the optimization stage and evaluates them on the same fixed test sample,
 * enabling a direct comparison of their ROC curves and AUC values.
 *
 * Part of the bachelor's thesis:
 * "Application of Machine Learning Techniques for Event Classification
 *  in ATLAS Open Data"
 *
 * Author: Pavlos Panagiotopoulos
 * National and Kapodistrian University of Athens
 * 2026
 */

#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TROOT.h"

#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/Types.h"


void compare_models_nj2plus()
{
    TMVA::Tools::Instance();

    std::cout << "\n========================================"
              << "\nFINAL TMVA CLASSIFIER COMPARISON -- nj >= 2"
              << "\n========================================"
              << std::endl;


    // ============================================================
    // Input datasets
    // ============================================================

    TFile* sigFile = TFile::Open("nj2_signal.root");
    TFile* bkgFile = TFile::Open("nj2_background.root");

    if (!sigFile || sigFile->IsZombie() ||
        !bkgFile || bkgFile->IsZombie()) {

        std::cerr << "ERROR opening input files." << std::endl;
        return;
    }

    TTree* sigTree =
        (TTree*)sigFile->Get("signal");

    TTree* bkgTree =
        (TTree*)bkgFile->Get("background");

    if (!sigTree || !bkgTree) {

        std::cerr << "ERROR loading input trees." << std::endl;

        sigFile->Close();
        bkgFile->Close();

        delete sigFile;
        delete bkgFile;

        return;
    }

    std::cout << "\nSignal entries     = "
              << sigTree->GetEntries()
              << "\nBackground entries = "
              << bkgTree->GetEntries()
              << std::endl;


    // ============================================================
    // Output file
    // ============================================================

    TFile* outputFile =
        TFile::Open(
            "comparison_nj2plus_results.root",
            "RECREATE");

    if (!outputFile || outputFile->IsZombie()) {

        std::cerr
            << "ERROR creating output file."
            << std::endl;

        sigFile->Close();
        bkgFile->Close();

        delete sigFile;
        delete bkgFile;

        return;
    }


    // ============================================================
    // Factory and DataLoader
    // ============================================================

    TMVA::Factory* factory =
        new TMVA::Factory(
            "Comparison_nj2plus",
            outputFile,
            "!V:!Silent:Color:DrawProgressBar:"
            "Transformations=I:"
            "AnalysisType=Classification");

    TMVA::DataLoader* loader =
        new TMVA::DataLoader(
            "dataset_Comparison_nj2plus");


    // ============================================================
    // 15 input variables
    // ============================================================

    loader->AddVariable("pT_mumu", 'F');
    loader->AddVariable("Y_mumu", 'F');
    loader->AddVariable("cosTheta", 'F');

    loader->AddVariable("jet1_pt", 'F');
    loader->AddVariable("jet1_eta", 'F');
    loader->AddVariable("dPhi_j1_mumu", 'F');

    loader->AddVariable("jet2_pt", 'F');
    loader->AddVariable("jet2_eta", 'F');
    loader->AddVariable("dPhi_j2_mumu", 'F');

    loader->AddVariable("pT_jj", 'F');
    loader->AddVariable("Y_jj", 'F');
    loader->AddVariable("dPhi_jj_mumu", 'F');

    loader->AddVariable("m_jj", 'F');
    loader->AddVariable("met", 'F');
    loader->AddVariable("HT", 'F');


    // ============================================================
    // Spectators
    // ============================================================

    loader->AddSpectator("m_mumu", 'F');
    loader->AddSpectator("eventNumber", 'l');
    loader->AddSpectator("jet_n", 'I');
    loader->AddSpectator("channelNumber", 'i');


    // ============================================================
    // Trees and physical event weights
    // ============================================================

    loader->AddSignalTree(sigTree, 1.0);
    loader->AddBackgroundTree(bkgTree, 1.0);

    loader->SetSignalWeightExpression("eventWeight");
    loader->SetBackgroundWeightExpression("eventWeight");


    // ============================================================
    // Common fixed 80/20 train/test split
    // ============================================================

    loader->PrepareTrainingAndTestTree("", "",
        "nTrain_Signal=34550:"
        "nTest_Signal=8638:"
        "nTrain_Background=100521:"
        "nTest_Background=25131:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");


    // ============================================================
    // DNN
    // Selected architecture from 5-fold CV: 30-30
    // ============================================================

    TString dnnTrainingStrategy =
        "TrainingStrategy="
        "LearningRate=1e-3,"
        "Momentum=0.9,"
        "ConvergenceSteps=50,"
        "BatchSize=100,"
        "TestRepetitions=1,"
        "MaxEpochs=1000,"
        "WeightDecay=0.0,"
        "Regularization=None,"
        "Optimizer=ADAM";

    TString dnnOptions =
        "!H:!V:"
        "ErrorStrategy=CROSSENTROPY:"
        "VarTransform=N:"
        "WeightInitialization=XAVIERUNIFORM:"
        "Architecture=CPU:"
        "Layout=TANH|30,TANH|30,LINEAR:"
        + dnnTrainingStrategy;

    factory->BookMethod(
        loader,
        TMVA::Types::kDL,
        "DNN",
        dnnOptions);


    // ============================================================
    // BDT
    // Selected MaxDepth from 5-fold CV: 2
    // ============================================================

    TString bdtOptions =
        "!H:!V:"
        "NTrees=800:"
        "MinNodeSize=2.5%:"
        "MaxDepth=2:"
        "BoostType=AdaBoost:"
        "AdaBoostBeta=0.5:"
        "UseBaggedBoost:"
        "BaggedSampleFraction=0.5:"
        "SeparationType=GiniIndex:"
        "nCuts=20:"
        "NegWeightTreatment=IgnoreNegWeightsInTraining";

    factory->BookMethod(
        loader,
        TMVA::Types::kBDT,
        "BDT",
        bdtOptions);


    // ============================================================
    // Training, testing and evaluation
    // ============================================================

    std::cout << "\n========================================"
              << "\nFINAL DNN + BDT COMPARISON -- nj >= 2"
              << "\n========================================"
              << "\nDNN architecture = 30-30"
              << "\nBDT MaxDepth     = 2"
              << "\n========================================"
              << std::endl;

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();


    // ============================================================
    // Common test AUC values
    // ============================================================

    double dnnAUC =
        factory->GetROCIntegral(
            loader,
            "DNN",
            0,
            TMVA::Types::kTesting);

    double bdtAUC =
        factory->GetROCIntegral(
            loader,
            "BDT",
            0,
            TMVA::Types::kTesting);


    // ============================================================
    // Save results
    // ============================================================

    outputFile->Write();
    outputFile->Close();

    delete factory;
    delete loader;


    // ============================================================
    // Native TMVA ROC comparison
    // ============================================================

    if (!gROOT->IsBatch())
        TMVA::TMVAGui(
            "comparison_nj2plus_results.root");


    // ============================================================
    // Final summary
    // ============================================================

    std::cout << "\n========================================"
              << "\nFINAL COMMON TEST AUC -- nj >= 2"
              << "\n========================================"
              << "\nDNN"
              << "\n Architecture = 30-30"
              << "\n Test AUC     = " << dnnAUC
              << "\n"
              << "\nBDT"
              << "\n MaxDepth     = 2"
              << "\n NTrees       = 800"
              << "\n Test AUC     = " << bdtAUC
              << "\n"
              << "\nSignal train      = 34550"
              << "\nSignal test       = 8638"
              << "\nBackground train  = 100521"
              << "\nBackground test   = 25131"
              << "\n"
              << "\nOutput ROOT file: comparison_nj2plus_results.root"
              << "\n========================================"
              << std::endl;


    // ============================================================
    // Cleanup
    // ============================================================

    sigFile->Close();
    bkgFile->Close();

    delete sigFile;
    delete bkgFile;
}