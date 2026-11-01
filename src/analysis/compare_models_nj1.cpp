/**
 * @file compare_models_nj1.cpp
 * @brief Trains and compares the final DNN and BDT classifiers
 *        for the n_j = 1 category.
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


void compare_models_nj1()
{
    TMVA::Tools::Instance();

    std::cout << "\n========================================"
              << "\nFINAL TMVA CLASSIFIER COMPARISON -- nj = 1"
              << "\n========================================"
              << std::endl;


    // ============================================================
    // Input datasets
    // ============================================================

    TFile* sigFile = TFile::Open("nj1_signal.root");
    TFile* bkgFile = TFile::Open("nj1_background.root");

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
            "comparison_nj1_results.root",
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
            "Comparison_nj1",
            outputFile,
            "!V:!Silent:Color:DrawProgressBar:"
            "Transformations=I:"
            "AnalysisType=Classification");

    TMVA::DataLoader* loader =
        new TMVA::DataLoader(
            "dataset_Comparison_nj1");


    // ============================================================
    // Input variables
    // ============================================================

    loader->AddVariable("pT_mumu", 'F');
    loader->AddVariable("Y_mumu", 'F');
    loader->AddVariable("cosTheta", 'F');

    loader->AddVariable("jet1_pt", 'F');
    loader->AddVariable("jet1_eta", 'F');
    loader->AddVariable("dPhi_mumu_j1", 'F');


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
        "nTrain_Signal=42378:"
        "nTest_Signal=10595:"
        "nTrain_Background=57632:"
        "nTest_Background=14408:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");


    // ============================================================
    // DNN
    // Selected architecture from 5-fold CV: 6-6-6
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
        "Layout=TANH|6,TANH|6,TANH|6,LINEAR:"
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
              << "\nFINAL DNN + BDT COMPARISON -- nj = 1"
              << "\n========================================"
              << "\nDNN architecture = 6-6-6"
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
            "comparison_nj1_results.root");


    // ============================================================
    // Final summary
    // ============================================================

    std::cout << "\n========================================"
              << "\nFINAL COMMON TEST AUC -- nj = 1"
              << "\n========================================"
              << "\nDNN"
              << "\n Architecture = 6-6-6"
              << "\n Test AUC     = " << dnnAUC
              << "\n"
              << "\nBDT"
              << "\n MaxDepth     = 2"
              << "\n NTrees       = 800"
              << "\n Test AUC     = " << bdtAUC
              << "\n"
              << "\nSignal train      = 42378"
              << "\nSignal test       = 10595"
              << "\nBackground train  = 57632"
              << "\nBackground test   = 14408"
              << "\n"
              << "\nOutput ROOT file: comparison_nj1_results.root"
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