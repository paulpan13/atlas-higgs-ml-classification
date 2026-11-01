/**
 * @file dnn.cpp
 * @brief Tunes, trains, and evaluates a Deep Neural Network classifier
 *        for the n_j = 1 category.
 *
 * Performs 5-fold cross-validation over a set of predefined neural-network
 * architectures, then trains the final DNN using the selected configuration
 * and evaluates it on the fixed test sample.
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
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/CrossValidation.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/Types.h"


void dnn()
{
    TMVA::Tools::Instance();


    // ============================================================
    // DNN configurations
    // ============================================================

   std::vector<TString> configNames = {
    "6_6",
    "12_12",
    "24_12",
    "6_6_6",
    "12_12_12",
    "24_24_24",
    "24_12_6",
    "12_12_12_12",
    "36_24_12_6"
};

std::vector<TString> layouts = {
    "Layout=TANH|6,TANH|6,LINEAR",
    "Layout=TANH|12,TANH|12,LINEAR",
    "Layout=TANH|24,TANH|12,LINEAR",
    "Layout=TANH|6,TANH|6,TANH|6,LINEAR",
    "Layout=TANH|12,TANH|12,TANH|12,LINEAR",
    "Layout=TANH|24,TANH|24,TANH|24,LINEAR",
    "Layout=TANH|24,TANH|12,TANH|6,LINEAR",
    "Layout=TANH|12,TANH|12,TANH|12,TANH|12,LINEAR",
    "Layout=TANH|36,TANH|24,TANH|12,TANH|6,LINEAR"
};

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

    TTree* sigTree = (TTree*)sigFile->Get("signal");
    TTree* bkgTree = (TTree*)bkgFile->Get("background");

    if (!sigTree || !bkgTree) {
        std::cerr << "ERROR loading input trees." << std::endl;
        sigFile->Close();
        bkgFile->Close();
        return;
    }

    std::cout << "\nSignal entries     = " << sigTree->GetEntries()
              << "\nBackground entries = " << bkgTree->GetEntries()
              << std::endl;


    // ============================================================
    // 1. 5-FOLD CROSS-VALIDATION
    // ============================================================

    TFile* cvFile =
        TFile::Open("DNN_nj1_CV_results.root", "RECREATE");

    if (!cvFile || cvFile->IsZombie()) {
        std::cerr << "ERROR creating CV output file." << std::endl;
        sigFile->Close();
        bkgFile->Close();
        return;
    }

    TMVA::DataLoader* cvLoader =
        new TMVA::DataLoader("cv_DNN_nj1");


    // ------------------------------------------------------------
    // Input variables
    // ------------------------------------------------------------

    cvLoader->AddVariable("pT_mumu", 'F');
    cvLoader->AddVariable("Y_mumu", 'F');
    cvLoader->AddVariable("cosTheta", 'F');

    cvLoader->AddVariable("jet1_pt", 'F');
    cvLoader->AddVariable("jet1_eta", 'F');
    cvLoader->AddVariable("dPhi_mumu_j1", 'F');


    // ------------------------------------------------------------
    // Spectators
    // ------------------------------------------------------------

    cvLoader->AddSpectator("m_mumu", 'F');
    cvLoader->AddSpectator("eventNumber", 'l');
    cvLoader->AddSpectator("jet_n", 'I');
    cvLoader->AddSpectator("channelNumber", 'i');


    // ------------------------------------------------------------
    // Trees and physical event weights
    // ------------------------------------------------------------

    cvLoader->AddSignalTree(sigTree, 1.0);
    cvLoader->AddBackgroundTree(bkgTree, 1.0);

    cvLoader->SetSignalWeightExpression("eventWeight");
    cvLoader->SetBackgroundWeightExpression("eventWeight");


    // ------------------------------------------------------------
    // 80% training pool / 20% untouched final test
    // ------------------------------------------------------------

    cvLoader->PrepareTrainingAndTestTree("", "",
        "nTrain_Signal=42378:"
        "nTest_Signal=10595:"
        "nTrain_Background=57632:"
        "nTest_Background=14408:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");


    double bestAUC = -1.0;
    double bestStd = 0.0;
    int bestConfig = -1;


    // ============================================================
    // Cross-validation
    // ============================================================

    {
        TString cvOptions =
            "!V:"
            "!Silent:"
            "!ModelPersistence:"
            "AnalysisType=Classification:"
            "SplitType=RandomStratified:"
            "NumFolds=5";

        TMVA::CrossValidation cv(
            "DNN_CV_nj1",
            cvLoader,
            cvFile,
            cvOptions);


        // --------------------------------------------------------
        // Fixed DNN training settings
        // --------------------------------------------------------

        TString trainingStrategy =
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

        TString commonOptions =
            "!H:!V:"
            "ErrorStrategy=CROSSENTROPY:"
            "VarTransform=N:"
            "WeightInitialization=XAVIERUNIFORM:"
            "Architecture=CPU";


        // --------------------------------------------------------
        // Book the nine configurations
        // --------------------------------------------------------

        for (size_t i = 0; i < layouts.size(); ++i) {

            TString methodName =
                "DNN_" + configNames[i];

            TString methodOptions =
                commonOptions + ":" +
                layouts[i] + ":" +
                trainingStrategy;

            cv.BookMethod(
                TMVA::Types::kDL,
                methodName,
                methodOptions);
        }


        // --------------------------------------------------------
        // Run CV
        // --------------------------------------------------------

        std::cout << "\n========================================"
                  << "\nDNN HYPERPARAMETER TUNING -- nj = 1"
                  << "\n5-fold cross-validation"
                  << "\n========================================"
                  << std::endl;

        cv.Evaluate();

        const auto& results = cv.GetResults();

        if (results.size() != layouts.size()) {
            std::cerr << "\nERROR: unexpected number of CV results."
                      << "\nExpected: " << layouts.size()
                      << "\nReceived: " << results.size()
                      << std::endl;
            return;
        }


        // --------------------------------------------------------
        // Detailed CV results
        // --------------------------------------------------------

        std::cout << "\n========================================"
                  << "\nDNN CROSS-VALIDATION RESULTS"
                  << "\n========================================"
                  << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {

            double meanAUC =
                results[i].GetROCAverage();

            double stdAUC =
                results[i].GetROCStandardDeviation();

            const auto& foldAUCs =
                results[i].GetROCValues();

            bool validConfiguration = true;

            std::cout << "\nConfiguration = "
                      << configNames[i]
                      << "\nFold AUCs:"
                      << std::endl;

            for (const auto& entry : foldAUCs) {

                std::cout << " Fold "
                          << entry.first + 1
                          << ": "
                          << entry.second;

                if (entry.second < 0.5) {
                    std::cout << "  <-- FAILED";
                    validConfiguration = false;
                }

                std::cout << std::endl;
            }

            std::cout << "Mean CV AUC = "
                      << meanAUC
                      << "\nCV AUC std  = "
                      << stdAUC
                      << std::endl;

            if (!validConfiguration) {

                std::cout
                    << "STATUS: INVALID configuration "
                    << "(at least one fold has AUC < 0.5)"
                    << std::endl;
            }
            else {

                std::cout
                    << "STATUS: VALID configuration"
                    << std::endl;

                if (meanAUC > bestAUC) {
                    bestAUC = meanAUC;
                    bestStd = stdAUC;
                    bestConfig = static_cast<int>(i);
                }
            }
        }


        // --------------------------------------------------------
        // Compact tuning summary
        // --------------------------------------------------------

        std::cout << "\n========================================"
                  << "\nDNN TUNING SUMMARY"
                  << "\n========================================"
                  << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {

            bool validConfiguration = true;

            const auto& foldAUCs =
                results[i].GetROCValues();

            for (const auto& entry : foldAUCs) {
                if (entry.second < 0.5) {
                    validConfiguration = false;
                    break;
                }
            }

            std::cout
                << "Configuration = "
                << configNames[i]
                << "   Mean AUC = "
                << results[i].GetROCAverage()
                << "   Std = "
                << results[i].GetROCStandardDeviation()
                << "   Status = "
                << (validConfiguration ? "VALID" : "INVALID")
                << std::endl;
        }


        // --------------------------------------------------------
        // Ensure a valid configuration exists
        // --------------------------------------------------------

        if (bestConfig < 0) {

            std::cerr
                << "\n========================================"
                << "\nERROR: NO VALID DNN CONFIGURATION"
                << "\nAt least one fold failed for every architecture."
                << "\nFinal DNN training will NOT be performed."
                << "\n========================================"
                << std::endl;

            cvFile->Write();
            cvFile->Close();

            delete cvLoader;

            sigFile->Close();
            bkgFile->Close();

            delete sigFile;
            delete bkgFile;

            return;
        }


        std::cout << "\n========================================"
                  << "\nSELECTED DNN CONFIGURATION"
                  << "\n========================================"
                  << "\nArchitecture = "
                  << configNames[bestConfig]
                  << "\nBest CV AUC  = "
                  << bestAUC
                  << "\nCV AUC std   = "
                  << bestStd
                  << "\n========================================"
                  << std::endl;
    }


    cvFile->Write();
    cvFile->Close();


    // ============================================================
    // 2. FINAL TRAINING
    // ============================================================

    if (gSystem->AccessPathName("dataset_DNN_nj1"))
        gSystem->mkdir("dataset_DNN_nj1");

    if (gSystem->AccessPathName("dataset_DNN_nj1")) {
        std::cerr
            << "ERROR creating dataset_DNN_nj1 directory."
            << std::endl;
        return;
    }

    TString outputPath =
        "dataset_DNN_nj1/DNN_results.root";

    TFile* outputFile =
        TFile::Open(outputPath, "RECREATE");

    if (!outputFile || outputFile->IsZombie()) {
        std::cerr
            << "ERROR creating final output file."
            << std::endl;
        return;
    }


    // ------------------------------------------------------------
    // Factory and final DataLoader
    // ------------------------------------------------------------

    TMVA::Factory* factory =
        new TMVA::Factory(
            "DNN",
            outputFile,
            "!V:!Silent:Color:DrawProgressBar:"
            "Transformations=I:"
            "AnalysisType=Classification");

    TMVA::DataLoader* finalLoader =
        new TMVA::DataLoader("dataset_DNN_nj1");


    // ------------------------------------------------------------
    // Same six input variables
    // ------------------------------------------------------------

    finalLoader->AddVariable("pT_mumu", 'F');
    finalLoader->AddVariable("Y_mumu", 'F');
    finalLoader->AddVariable("cosTheta", 'F');

    finalLoader->AddVariable("jet1_pt", 'F');
    finalLoader->AddVariable("jet1_eta", 'F');
    finalLoader->AddVariable("dPhi_mumu_j1", 'F');


    // ------------------------------------------------------------
    // Spectators
    // ------------------------------------------------------------

    finalLoader->AddSpectator("m_mumu", 'F');
    finalLoader->AddSpectator("eventNumber", 'l');
    finalLoader->AddSpectator("jet_n", 'I');
    finalLoader->AddSpectator("channelNumber", 'i');


    // ------------------------------------------------------------
    // Trees and weights
    // ------------------------------------------------------------

    finalLoader->AddSignalTree(sigTree, 1.0);
    finalLoader->AddBackgroundTree(bkgTree, 1.0);

    finalLoader->SetSignalWeightExpression("eventWeight");
    finalLoader->SetBackgroundWeightExpression("eventWeight");


    // ------------------------------------------------------------
    // Exact same 80/20 split
    // ------------------------------------------------------------

    finalLoader->PrepareTrainingAndTestTree("", "",
        "nTrain_Signal=42378:"
        "nTest_Signal=10595:"
        "nTrain_Background=57632:"
        "nTest_Background=14408:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");


    // ------------------------------------------------------------
    // Fixed settings
    // ------------------------------------------------------------

    TString trainingStrategy =
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

    TString commonOptions =
        "!H:!V:"
        "ErrorStrategy=CROSSENTROPY:"
        "VarTransform=N:"
        "WeightInitialization=XAVIERUNIFORM:"
        "Architecture=CPU";

    TString finalOptions =
        commonOptions + ":" +
        layouts[bestConfig] + ":" +
        trainingStrategy;


    // ------------------------------------------------------------
    // Book selected architecture
    // ------------------------------------------------------------

    factory->BookMethod(
        finalLoader,
        TMVA::Types::kDL,
        "DNN",
        finalOptions);


    std::cout << "\n========================================"
              << "\nFINAL DNN TRAINING -- nj = 1"
              << "\nSelected architecture = "
              << configNames[bestConfig]
              << "\n========================================"
              << std::endl;


    // ============================================================
    // Train / test / evaluate
    // ============================================================

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();


    // ============================================================
    // Final untouched-test AUC
    // ============================================================

    double testAUC =
        factory->GetROCIntegral(
            finalLoader,
            "DNN",
            0,
            TMVA::Types::kTesting);


    // ============================================================
    // Save
    // ============================================================

outputFile->Write();
outputFile->Close();

delete factory;
delete finalLoader;

// ============================================================
// 3. NATIVE TMVA PLOTS
// ============================================================

if (!gROOT->IsBatch())
    TMVA::TMVAGui("dataset_DNN_nj1/DNN_results.root");

// ============================================================
// 4. FINAL SUMMARY
// ============================================================

std::cout << "\n========================================"
          << "\nDNN FINAL SUMMARY -- nj = 1"
          << "\n========================================"
          << "\nSelected architecture = " << configNames[bestConfig]
          << "\nCV AUC                = " << bestAUC
          << "\nCV AUC std            = " << bestStd
          << "\nFinal test AUC        = " << testAUC
          << "\n"
          << "\nSignal train          = 42378"
          << "\nSignal test           = 10595"
          << "\nBackground train      = 57632"
          << "\nBackground test       = 14408"
          << "\n"
          << "\nOutput ROOT file      = dataset_DNN_nj1/DNN_results.root"
          << "\n========================================"
          << std::endl;

delete cvLoader;

sigFile->Close();
bkgFile->Close();

delete sigFile;
delete bkgFile;
}