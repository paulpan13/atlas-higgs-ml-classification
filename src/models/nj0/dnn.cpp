/**
 * @file dnn.cpp
 * @brief Tunes, trains, and evaluates a Deep Neural Network classifier
 *        for the n_j = 0 category.
 *
 * Performs 5-fold cross-validation to optimize the number of hidden
 * layers while keeping six neurons per hidden layer, then trains the
 * final classifier using the selected architecture and evaluates it
 * on the fixed test sample.
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

#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/CrossValidation.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/Types.h"

void dnn()
{
    TMVA::Tools::Instance();

    std::vector<int> depthValues = {1, 2, 3, 4, 5, 6};

    TFile* sigFile = TFile::Open("nj0_signal.root");
    TFile* bkgFile = TFile::Open("nj0_background.root");

    if (!sigFile || sigFile->IsZombie() || !bkgFile || bkgFile->IsZombie()) {
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

    TFile* cvFile = TFile::Open("DNN_CV_results.root", "RECREATE");

    if (!cvFile || cvFile->IsZombie()) {
        std::cerr << "ERROR creating CV output file." << std::endl;
        sigFile->Close();
        bkgFile->Close();
        return;
    }

    TMVA::DataLoader* cvLoader = new TMVA::DataLoader("cv_DNN");

    cvLoader->AddVariable("pT_mumu", 'F');
    cvLoader->AddVariable("Y_mumu", 'F');
    cvLoader->AddVariable("cosTheta", 'F');

    cvLoader->AddSpectator("m_mumu", 'F');
    cvLoader->AddSpectator("eventNumber", 'l');
    cvLoader->AddSpectator("jet_n", 'I');

    cvLoader->AddSignalTree(sigTree, 1.0);
    cvLoader->AddBackgroundTree(bkgTree, 1.0);

    cvLoader->SetSignalWeightExpression("eventWeight");
    cvLoader->SetBackgroundWeightExpression("eventWeight");

    cvLoader->PrepareTrainingAndTestTree("", "",
        "nTrain_Signal=46273:"
        "nTest_Signal=11569:"
        "nTrain_Background=31770:"
        "nTest_Background=7943:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");

    double bestAUC = -1.0;
    double bestStd = 0.0;
    int bestDepth = -1;

    {
        TString cvOptions =
            "!V:"
            "!Silent:"
            "!ModelPersistence:"
            "AnalysisType=Classification:"
            "SplitType=RandomStratified:"
            "NumFolds=5";

        TMVA::CrossValidation cv(
            "DNN_CV",
            cvLoader,
            cvFile,
            cvOptions);

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

        for (int depth : depthValues) {
            TString methodName = Form("DNN_L%d", depth);
            TString layout = "Layout=";

            for (int l = 0; l < depth; ++l) {
                if (l > 0) layout += ",";
                layout += "TANH|6";
            }

            layout += ",LINEAR";

            TString methodOptions =
                commonOptions + ":" +
                layout + ":" +
                trainingStrategy;

            cv.BookMethod(
                TMVA::Types::kDL,
                methodName,
                methodOptions);
        }

        std::cout << "\n========================================"
                  << "\nDNN HYPERPARAMETER TUNING"
                  << "\n5-fold cross-validation"
                  << "\n========================================"
                  << std::endl;

        cv.Evaluate();

        const auto& results = cv.GetResults();

        if (results.size() != depthValues.size()) {
            std::cerr << "\nERROR: unexpected number of CV results."
                      << "\nExpected: " << depthValues.size()
                      << "\nReceived: " << results.size()
                      << std::endl;
            return;
        }

        std::cout << "\n========================================"
                  << "\nDNN CROSS-VALIDATION RESULTS"
                  << "\n========================================"
                  << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {
            int depth = depthValues[i];

            double meanAUC = results[i].GetROCAverage();
            double stdAUC = results[i].GetROCStandardDeviation();
            const auto& foldAUCs = results[i].GetROCValues();

            bool validConfiguration = true;

            std::cout << "\nHidden layers = " << depth
                      << "\nFold AUCs:" << std::endl;

            for (const auto& entry : foldAUCs) {
                std::cout << " Fold " << entry.first + 1
                          << ": " << entry.second;

                if (entry.second < 0.5) {
                    std::cout << "  <-- FAILED";
                    validConfiguration = false;
                }

                std::cout << std::endl;
            }

            std::cout << "Mean CV AUC = " << meanAUC
                      << "\nCV AUC std  = " << stdAUC
                      << std::endl;

            if (!validConfiguration) {
                std::cout << "STATUS: INVALID configuration "
                          << "(at least one fold has AUC < 0.5)"
                          << std::endl;
            }
            else {
                std::cout << "STATUS: VALID configuration"
                          << std::endl;

                if (meanAUC > bestAUC) {
                    bestAUC = meanAUC;
                    bestStd = stdAUC;
                    bestDepth = depth;
                }
            }
        }

        std::cout << "\n========================================"
                  << "\nDNN TUNING SUMMARY"
                  << "\n========================================"
                  << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {
            bool validConfiguration = true;
            const auto& foldAUCs = results[i].GetROCValues();

            for (const auto& entry : foldAUCs) {
                if (entry.second < 0.5) {
                    validConfiguration = false;
                    break;
                }
            }

            std::cout << "Hidden layers = " << depthValues[i]
                      << "   Mean AUC = " << results[i].GetROCAverage()
                      << "   Std = " << results[i].GetROCStandardDeviation()
                      << "   Status = "
                      << (validConfiguration ? "VALID" : "INVALID")
                      << std::endl;
        }

        if (bestDepth < 0) {
            std::cerr << "\n========================================"
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

        std::cout << "\nSelected hidden layers = " << bestDepth
                  << "\nBest CV AUC            = " << bestAUC
                  << "\nCV AUC std             = " << bestStd
                  << "\n========================================"
                  << std::endl;
    }

    cvFile->Write();
    cvFile->Close();

    // ============================================================
    // 2. FINAL TRAINING
    // ============================================================

    if (gSystem->AccessPathName("dataset_DNN"))
        gSystem->mkdir("dataset_DNN");

    if (gSystem->AccessPathName("dataset_DNN")) {
        std::cerr << "ERROR creating dataset_DNN directory." << std::endl;
        return;
    }

    TString outputPath = "dataset_DNN/DNN_results.root";
    TFile* outputFile = TFile::Open(outputPath, "RECREATE");

    if (!outputFile || outputFile->IsZombie()) {
        std::cerr << "ERROR creating final output file." << std::endl;
        return;
    }

    TMVA::Factory* factory = new TMVA::Factory(
        "DNN",
        outputFile,
        "!V:!Silent:Color:DrawProgressBar:"
        "Transformations=I:"
        "AnalysisType=Classification");

    TMVA::DataLoader* finalLoader =
        new TMVA::DataLoader("dataset_DNN");

    finalLoader->AddVariable("pT_mumu", 'F');
    finalLoader->AddVariable("Y_mumu", 'F');
    finalLoader->AddVariable("cosTheta", 'F');

    finalLoader->AddSpectator("m_mumu", 'F');
    finalLoader->AddSpectator("eventNumber", 'l');
    finalLoader->AddSpectator("jet_n", 'I');

    finalLoader->AddSignalTree(sigTree, 1.0);
    finalLoader->AddBackgroundTree(bkgTree, 1.0);

    finalLoader->SetSignalWeightExpression("eventWeight");
    finalLoader->SetBackgroundWeightExpression("eventWeight");

    finalLoader->PrepareTrainingAndTestTree("", "",
        "nTrain_Signal=46273:"
        "nTest_Signal=11569:"
        "nTrain_Background=31770:"
        "nTest_Background=7943:"
        "SplitMode=Random:"
        "SplitSeed=12345:"
        "NormMode=NumEvents:"
        "!V");

    TString bestLayout = "Layout=";

    for (int l = 0; l < bestDepth; ++l) {
        if (l > 0) bestLayout += ",";
        bestLayout += "TANH|6";
    }

    bestLayout += ",LINEAR";

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
        bestLayout + ":" +
        trainingStrategy;

    factory->BookMethod(
        finalLoader,
        TMVA::Types::kDL,
        "DNN",
        finalOptions);

    std::cout << "\n========================================"
              << "\nFINAL DNN TRAINING"
              << "\nSelected hidden layers = " << bestDepth
              << "\nNeurons per hidden layer = 6"
              << "\n========================================"
              << std::endl;

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    double testAUC = factory->GetROCIntegral(
        finalLoader,
        "DNN",
        0,
        TMVA::Types::kTesting);

    std::cout << "\nFinal test AUC = " << testAUC << std::endl;
    
outputFile->Write();
outputFile->Close();

delete factory;
delete finalLoader;

// ============================================================
// 3. NATIVE TMVA PLOTS
// ============================================================

if (!gROOT->IsBatch())
    TMVA::TMVAGui("dataset_DNN/DNN_results.root");

// ============================================================
// 4. FINAL SUMMARY
// ============================================================

std::cout << "\n========================================"
          << "\nDNN FINAL SUMMARY"
          << "\n========================================"
          << "\nSelected hidden layers = " << bestDepth
          << "\nNeurons per layer       = 6"
          << "\nCV AUC                  = " << bestAUC
          << "\nCV AUC std              = " << bestStd
          << "\nFinal test AUC          = " << testAUC
          << "\n"
          << "\nSignal train            = 46273"
          << "\nSignal test             = 11569"
          << "\nBackground train        = 31770"
          << "\nBackground test         = 7943"
          << "\n"
          << "\nOutput ROOT file        = dataset_DNN/DNN_results.root"
          << "\n========================================"
          << std::endl;

delete cvLoader;

sigFile->Close();
bkgFile->Close();

delete sigFile;
delete bkgFile;
}