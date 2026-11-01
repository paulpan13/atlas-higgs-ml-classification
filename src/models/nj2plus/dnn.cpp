/**
 * @file dnn.cpp
 * @brief Tunes, trains, and evaluates a Deep Neural Network classifier
 *        for the n_j >= 2 category.
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
    // DNN architectures
    // ============================================================

    std::vector<TString> configNames = {
        "15_15",
        "30_15",
        "30_30",
        "15_15_15",
        "30_30_30",
        "60_30_15",
        "15_15_15_15",
        "30_30_30_30",
        "75_60_30_15"
    };

    std::vector<TString> layouts = {
        "Layout=TANH|15,TANH|15,LINEAR",
        "Layout=TANH|30,TANH|15,LINEAR",
        "Layout=TANH|30,TANH|30,LINEAR",
        "Layout=TANH|15,TANH|15,TANH|15,LINEAR",
        "Layout=TANH|30,TANH|30,TANH|30,LINEAR",
        "Layout=TANH|60,TANH|30,TANH|15,LINEAR",
        "Layout=TANH|15,TANH|15,TANH|15,TANH|15,LINEAR",
        "Layout=TANH|30,TANH|30,TANH|30,TANH|30,LINEAR",
        "Layout=TANH|75,TANH|60,TANH|30,TANH|15,LINEAR"
    };


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
        TFile::Open("DNN_nj2_CV_results.root", "RECREATE");

    if (!cvFile || cvFile->IsZombie()) {
        std::cerr << "ERROR creating CV output file." << std::endl;
        sigFile->Close();
        bkgFile->Close();
        return;
    }

    TMVA::DataLoader* cvLoader =
        new TMVA::DataLoader("cv_DNN_nj2");


    // ============================================================
    // 15 input variables
    // ============================================================

    cvLoader->AddVariable("pT_mumu", 'F');
    cvLoader->AddVariable("Y_mumu", 'F');
    cvLoader->AddVariable("cosTheta", 'F');

    cvLoader->AddVariable("jet1_pt", 'F');
    cvLoader->AddVariable("jet1_eta", 'F');
    cvLoader->AddVariable("dPhi_j1_mumu", 'F');

    cvLoader->AddVariable("jet2_pt", 'F');
    cvLoader->AddVariable("jet2_eta", 'F');
    cvLoader->AddVariable("dPhi_j2_mumu", 'F');

    cvLoader->AddVariable("pT_jj", 'F');
    cvLoader->AddVariable("Y_jj", 'F');
    cvLoader->AddVariable("dPhi_jj_mumu", 'F');

    cvLoader->AddVariable("m_jj", 'F');
    cvLoader->AddVariable("met", 'F');
    cvLoader->AddVariable("HT", 'F');


    // ============================================================
    // Spectators
    // ============================================================

    cvLoader->AddSpectator("m_mumu", 'F');
    cvLoader->AddSpectator("eventNumber", 'l');
    cvLoader->AddSpectator("jet_n", 'I');
    cvLoader->AddSpectator("channelNumber", 'i');


    // ============================================================
    // Trees and physical event weights
    // ============================================================

    cvLoader->AddSignalTree(sigTree, 1.0);
    cvLoader->AddBackgroundTree(bkgTree, 1.0);

    cvLoader->SetSignalWeightExpression("eventWeight");
    cvLoader->SetBackgroundWeightExpression("eventWeight");


    // ============================================================
    // 80% training / 20% final test
    // ============================================================

    cvLoader->PrepareTrainingAndTestTree("", "",
        "nTrain_Signal=34550:"
        "nTest_Signal=8638:"
        "nTrain_Background=100521:"
        "nTest_Background=25131:"
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
            "DNN_CV_nj2",
            cvLoader,
            cvFile,
            cvOptions);


        // --------------------------------------------------------
        // Fixed DNN settings
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
        // Book architectures
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
                  << "\nDNN HYPERPARAMETER TUNING -- nj >= 2"
                  << "\n5-fold cross-validation"
                  << "\n========================================"
                  << std::endl;

        cv.Evaluate();

        const auto& results = cv.GetResults();

        if (results.size() != layouts.size()) {

            std::cerr
                << "\nERROR: unexpected number of CV results."
                << "\nExpected: " << layouts.size()
                << "\nReceived: " << results.size()
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


        // ========================================================
        // Detailed CV results
        // ========================================================

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

            std::cout << "\nArchitecture = "
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


        // ========================================================
        // Compact tuning summary
        // ========================================================

        std::cout << "\n========================================"
                  << "\nDNN TUNING SUMMARY -- nj >= 2"
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
                << "Architecture = "
                << configNames[i]
                << "   Mean AUC = "
                << results[i].GetROCAverage()
                << "   Std = "
                << results[i].GetROCStandardDeviation()
                << "   Status = "
                << (validConfiguration ? "VALID" : "INVALID")
                << std::endl;
        }


        // ========================================================
        // Ensure valid configuration exists
        // ========================================================

        if (bestConfig < 0) {

            std::cerr
                << "\n========================================"
                << "\nERROR: NO VALID DNN CONFIGURATION"
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

    if (gSystem->AccessPathName("dataset_DNN_nj2"))
        gSystem->mkdir("dataset_DNN_nj2");

    if (gSystem->AccessPathName("dataset_DNN_nj2")) {
        std::cerr
            << "ERROR creating dataset_DNN_nj2 directory."
            << std::endl;
        return;
    }

    TString outputPath =
        "dataset_DNN_nj2/DNN_results.root";

    TFile* outputFile =
        TFile::Open(outputPath, "RECREATE");

    if (!outputFile || outputFile->IsZombie()) {
        std::cerr
            << "ERROR creating final output file."
            << std::endl;
        return;
    }


    // ============================================================
    // Factory and final DataLoader
    // ============================================================

    TMVA::Factory* factory =
        new TMVA::Factory(
            "DNN_nj2",
            outputFile,
            "!V:!Silent:Color:DrawProgressBar:"
            "Transformations=I:"
            "AnalysisType=Classification");

    TMVA::DataLoader* finalLoader =
        new TMVA::DataLoader("dataset_DNN_nj2");


    // ============================================================
    // Same 15 input variables
    // ============================================================

    finalLoader->AddVariable("pT_mumu", 'F');
    finalLoader->AddVariable("Y_mumu", 'F');
    finalLoader->AddVariable("cosTheta", 'F');

    finalLoader->AddVariable("jet1_pt", 'F');
    finalLoader->AddVariable("jet1_eta", 'F');
    finalLoader->AddVariable("dPhi_j1_mumu", 'F');

    finalLoader->AddVariable("jet2_pt", 'F');
    finalLoader->AddVariable("jet2_eta", 'F');
    finalLoader->AddVariable("dPhi_j2_mumu", 'F');

    finalLoader->AddVariable("pT_jj", 'F');
    finalLoader->AddVariable("Y_jj", 'F');
    finalLoader->AddVariable("dPhi_jj_mumu", 'F');

    finalLoader->AddVariable("m_jj", 'F');
    finalLoader->AddVariable("met", 'F');
    finalLoader->AddVariable("HT", 'F');


    // ============================================================
    // Spectators
    // ============================================================

    finalLoader->AddSpectator("m_mumu", 'F');
    finalLoader->AddSpectator("eventNumber", 'l');
    finalLoader->AddSpectator("jet_n", 'I');
    finalLoader->AddSpectator("channelNumber", 'i');


    // ============================================================
    // Trees and event weights
    // ============================================================

    finalLoader->AddSignalTree(sigTree, 1.0);
    finalLoader->AddBackgroundTree(bkgTree, 1.0);

    finalLoader->SetSignalWeightExpression("eventWeight");
    finalLoader->SetBackgroundWeightExpression("eventWeight");


    // ============================================================
    // Same 80/20 split
    // ============================================================

    finalLoader->PrepareTrainingAndTestTree("", "",
        "nTrain_Signal=34550:"
        "nTest_Signal=8638:"
        "nTrain_Background=100521:"
        "nTest_Background=25131:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");


    // ============================================================
    // Fixed DNN settings
    // ============================================================

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


    // ============================================================
    // Book selected architecture
    // ============================================================

    factory->BookMethod(
        finalLoader,
        TMVA::Types::kDL,
        "DNN",
        finalOptions);


    std::cout << "\n========================================"
              << "\nFINAL DNN TRAINING -- nj >= 2"
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
    TMVA::TMVAGui("dataset_DNN_nj2/DNN_results.root");

// ============================================================
// 4. FINAL SUMMARY
// ============================================================

std::cout << "\n========================================"
          << "\nDNN FINAL SUMMARY -- nj >= 2"
          << "\n========================================"
          << "\nSelected architecture = " << configNames[bestConfig]
          << "\nCV AUC                = " << bestAUC
          << "\nCV AUC std            = " << bestStd
          << "\nFinal test AUC        = " << testAUC
          << "\n"
          << "\nSignal train          = 34550"
          << "\nSignal test           = 8638"
          << "\nBackground train      = 100521"
          << "\nBackground test       = 25131"
          << "\n"
          << "\nOutput ROOT file      = dataset_DNN_nj2/DNN_results.root"
          << "\n========================================"
          << std::endl;

delete cvLoader;

sigFile->Close();
bkgFile->Close();

delete sigFile;
delete bkgFile;
}