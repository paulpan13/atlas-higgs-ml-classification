/**
 * @file fisher.cpp
 * @brief Trains and evaluates a Fisher classifier for the n_j = 0 category.
 *
 * Uses the final n_j = 0 signal and background datasets and the three
 * dimuon input variables pT_mumu, Y_mumu, and cosTheta. Event weights
 * are included during training and evaluation.
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

#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Types.h"

void fisher()
{
    TMVA::Tools::Instance();

    TFile* sigFile = TFile::Open("nj0_signal.root");
    TFile* bkgFile = TFile::Open("nj0_background.root");

    TTree* sigTree = (TTree*)sigFile->Get("signal");
    TTree* bkgTree = (TTree*)bkgFile->Get("background");

    TFile* outputFile =
        TFile::Open("Fisher_results.root", "RECREATE");

    TMVA::Factory* factory = new TMVA::Factory(
        "Fisher",
        outputFile,
        "!V:!Silent:Color:DrawProgressBar:"
        "Transformations=I:"
        "AnalysisType=Classification");

    TMVA::DataLoader* loader =
        new TMVA::DataLoader("dataset_Fisher");

    // Input variables
    loader->AddVariable("pT_mumu", 'F');
    loader->AddVariable("Y_mumu", 'F');
    loader->AddVariable("cosTheta", 'F');

    // Spectator variables
    loader->AddSpectator("m_mumu", 'F');
    loader->AddSpectator("eventNumber", 'l');
    loader->AddSpectator("jet_n", 'I');

    // Signal and background samples
    loader->AddSignalTree(sigTree, 1.0);
    loader->AddBackgroundTree(bkgTree, 1.0);

    // Physical event weights
    loader->SetSignalWeightExpression("eventWeight");
    loader->SetBackgroundWeightExpression("eventWeight");

    // Fixed 80/20 training-test split
    loader->PrepareTrainingAndTestTree("", "",
        "nTrain_Signal=46273:"
        "nTest_Signal=11569:"
        "nTrain_Background=31770:"
        "nTest_Background=7943:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");

    // Fisher classifier
    factory->BookMethod(
        loader,
        TMVA::Types::kFisher,
        "Fisher",
        "!H:!V:"
        "Fisher:"
        "VarTransform=None:"
        "IgnoreNegWeightsInTraining");

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    double testAUC = factory->GetROCIntegral(
        loader,
        "Fisher",
        0,
        TMVA::Types::kTesting);

    std::cout << "Final test AUC = "
              << testAUC << std::endl;

    outputFile->Write();
    outputFile->Close();

    delete factory;
    delete loader;

    sigFile->Close();
    bkgFile->Close();
}