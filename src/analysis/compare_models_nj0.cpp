/**
 * @file compare_models_nj0.cpp
 * @brief Trains and compares the final TMVA classifiers for the n_j = 0 category.
 *
 * Retrains the six classifiers using the hyperparameters selected during
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


void compare_models_nj0()
{
    TMVA::Tools::Instance();

    std::cout << "\n========================================"
              << "\nFINAL TMVA CLASSIFIER COMPARISON -- nj = 0"
              << "\n========================================"
              << std::endl;


    // ============================================================
    // Input datasets
    // ============================================================

    TFile* sigFile = TFile::Open("nj0_signal.root");
    TFile* bkgFile = TFile::Open("nj0_background.root");

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
            "comparison_nj0_results.root",
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
            "Comparison_nj0",
            outputFile,
            "!V:!Silent:Color:DrawProgressBar:"
            "Transformations=I:"
            "AnalysisType=Classification");

    TMVA::DataLoader* loader =
        new TMVA::DataLoader(
            "dataset_Comparison_nj0");


    // ============================================================
    // Input variables
    // ============================================================

    loader->AddVariable("pT_mumu", 'F');
    loader->AddVariable("Y_mumu", 'F');
    loader->AddVariable("cosTheta", 'F');


    // ============================================================
    // Spectators
    // ============================================================

    loader->AddSpectator("m_mumu", 'F');
    loader->AddSpectator("eventNumber", 'l');
    loader->AddSpectator("jet_n", 'I');


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
        "nTrain_Signal=46273:"
        "nTest_Signal=11569:"
        "nTrain_Background=31770:"
        "nTest_Background=7943:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");


    // ============================================================
    // FISHER
    // ============================================================

    factory->BookMethod(
        loader,
        TMVA::Types::kFisher,
        "Fisher",
        "!H:!V:"
        "Fisher:"
        "VarTransform=None:"
        "IgnoreNegWeightsInTraining");


    // ============================================================
    // LIKELIHOOD
    // Selected: Nbins = 150
    // ============================================================

    factory->BookMethod(
        loader,
        TMVA::Types::kLikelihood,
        "Likelihood",
        "!H:!V:"
        "TransformOutput=false:"
        "PDFInterpol=Spline2:"
        "NSmooth=1:"
        "NAvEvtPerBin=0:"
        "Nbins=150:"
        "VarTransform=None:"
        "IgnoreNegWeightsInTraining");


    // ============================================================
    // kNN
    // Selected: k = 300
    // ============================================================

    factory->BookMethod(
        loader,
        TMVA::Types::kKNN,
        "KNN",
        "!H:!V:"
        "nkNN=300:"
        "ScaleFrac=0.8:"
        "SigmaFact=1.0:"
        "Kernel=Gaus:"
        "UseKernel=F:"
        "UseWeight=T:"
        "Trim=F:"
        "IgnoreNegWeightsInTraining");


    // ============================================================
    // MLP
    // Selected: 14 hidden neurons
    // ============================================================

    factory->BookMethod(
        loader,
        TMVA::Types::kMLP,
        "MLP",
        "!H:!V:"
        "NeuronType=tanh:"
        "VarTransform=N:"
        "NCycles=600:"
        "HiddenLayers=14:"
        "TestRate=5:"
        "TrainingMethod=BP:"
        "!UseRegulator:"
        "IgnoreNegWeightsInTraining");


    // ============================================================
    // DNN
    // Selected: 5 hidden layers, 6 neurons per layer
    // ============================================================

    TString dnnOptions =
        "!H:!V:"
        "ErrorStrategy=CROSSENTROPY:"
        "VarTransform=N:"
        "WeightInitialization=XAVIERUNIFORM:"
        "Architecture=CPU:"
        "Layout=TANH|6,TANH|6,TANH|6,TANH|6,TANH|6,LINEAR:"
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

    factory->BookMethod(
        loader,
        TMVA::Types::kDL,
        "DNN",
        dnnOptions);


    // ============================================================
    // BDT
    // Selected: MaxDepth = 2
    // ============================================================

    factory->BookMethod(
        loader,
        TMVA::Types::kBDT,
        "BDT",
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
        "NegWeightTreatment=IgnoreNegWeightsInTraining");


    // ============================================================
    // Training, testing and evaluation
    // ============================================================

    std::cout
        << "\n==> Training all classifiers..."
        << std::endl;

    factory->TrainAllMethods();

    std::cout
        << "\n==> Testing all classifiers..."
        << std::endl;

    factory->TestAllMethods();

    std::cout
        << "\n==> Evaluating all classifiers..."
        << std::endl;

    factory->EvaluateAllMethods();


    // ============================================================
    // Common test AUC values
    // ============================================================

    double aucFisher =
        factory->GetROCIntegral(
            loader,
            "Fisher",
            0,
            TMVA::Types::kTesting);

    double aucLikelihood =
        factory->GetROCIntegral(
            loader,
            "Likelihood",
            0,
            TMVA::Types::kTesting);

    double aucKNN =
        factory->GetROCIntegral(
            loader,
            "KNN",
            0,
            TMVA::Types::kTesting);

    double aucMLP =
        factory->GetROCIntegral(
            loader,
            "MLP",
            0,
            TMVA::Types::kTesting);

    double aucDNN =
        factory->GetROCIntegral(
            loader,
            "DNN",
            0,
            TMVA::Types::kTesting);

    double aucBDT =
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
            "comparison_nj0_results.root");


    // ============================================================
    // Final summary
    // ============================================================

    std::cout
        << "\n========================================"
        << "\nFINAL COMMON TEST AUC -- nj = 0"
        << "\n========================================"
        << "\nFisher     : " << aucFisher
        << "\nLikelihood : " << aucLikelihood
        << "\nKNN        : " << aucKNN
        << "\nMLP        : " << aucMLP
        << "\nDNN        : " << aucDNN
        << "\nBDT        : " << aucBDT
        << "\n"
        << "\nOutput ROOT file: comparison_nj0_results.root"
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