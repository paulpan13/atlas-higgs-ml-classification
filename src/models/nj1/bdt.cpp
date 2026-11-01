/**
 * @file bdt.cpp
 * @brief Tunes, trains, and evaluates a Boosted Decision Tree classifier
 *        for the n_j = 1 category.
 *
 * Performs 5-fold cross-validation to optimize the maximum tree depth,
 * then trains the final BDT using the selected configuration and
 * evaluates it on the fixed test sample.
 *
 * Part of the bachelor's thesis:
 * "Application of Machine Learning Techniques for Event Classification
 *  in ATLAS Open Data"
 *
 * Author: Pavlos Panagiotoulos
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


void bdt()
{
    TMVA::Tools::Instance();


    // ============================================================
    // MaxDepth values
    // ============================================================

    std::vector<int> depthValues = {1, 2, 3, 4, 5, 6, 7, 8};


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
        TFile::Open("BDT_nj1_CV_results.root", "RECREATE");

    if (!cvFile || cvFile->IsZombie()) {
        std::cerr << "ERROR creating CV output file." << std::endl;
        sigFile->Close();
        bkgFile->Close();
        return;
    }

    TMVA::DataLoader* cvLoader =
        new TMVA::DataLoader("cv_BDT_nj1");


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
    // Trees and event weights
    // ------------------------------------------------------------

    cvLoader->AddSignalTree(sigTree, 1.0);
    cvLoader->AddBackgroundTree(bkgTree, 1.0);

    cvLoader->SetSignalWeightExpression("eventWeight");
    cvLoader->SetBackgroundWeightExpression("eventWeight");


    // ------------------------------------------------------------
    // 80% training / 20% final test
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
    int bestDepth = -1;


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
            "BDT_CV_nj1",
            cvLoader,
            cvFile,
            cvOptions);


        // --------------------------------------------------------
        // Fixed BDT settings
        // --------------------------------------------------------

        TString commonOptions =
            "!H:!V:"
            "NTrees=800:"
            "MinNodeSize=2.5%:"
            "BoostType=AdaBoost:"
            "AdaBoostBeta=0.5:"
            "UseBaggedBoost:"
            "BaggedSampleFraction=0.5:"
            "SeparationType=GiniIndex:"
            "nCuts=20:"
            "NegWeightTreatment=IgnoreNegWeightsInTraining";


        // --------------------------------------------------------
        // Book MaxDepth configurations
        // --------------------------------------------------------

        for (int depth : depthValues) {

            TString methodName =
                Form("BDT_D%d", depth);

            TString methodOptions =
                commonOptions +
                Form(":MaxDepth=%d", depth);

            cv.BookMethod(
                TMVA::Types::kBDT,
                methodName,
                methodOptions);
        }


        // --------------------------------------------------------
        // Run CV
        // --------------------------------------------------------

        std::cout << "\n========================================"
                  << "\nBDT HYPERPARAMETER TUNING -- nj = 1"
                  << "\n5-fold cross-validation"
                  << "\n========================================"
                  << std::endl;

        cv.Evaluate();

        const auto& results = cv.GetResults();

        if (results.size() != depthValues.size()) {
            std::cerr
                << "\nERROR: unexpected number of CV results."
                << "\nExpected: " << depthValues.size()
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


        // --------------------------------------------------------
        // Detailed CV results
        // --------------------------------------------------------

        std::cout << "\n========================================"
                  << "\nBDT CROSS-VALIDATION RESULTS"
                  << "\n========================================"
                  << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {

            int depth = depthValues[i];

            double meanAUC =
                results[i].GetROCAverage();

            double stdAUC =
                results[i].GetROCStandardDeviation();

            const auto& foldAUCs =
                results[i].GetROCValues();

            std::cout << "\nMaxDepth = " << depth
                      << "\nFold AUCs:"
                      << std::endl;

            for (const auto& entry : foldAUCs) {
                std::cout << " Fold "
                          << entry.first + 1
                          << ": "
                          << entry.second
                          << std::endl;
            }

            std::cout << "Mean CV AUC = "
                      << meanAUC
                      << "\nCV AUC std  = "
                      << stdAUC
                      << std::endl;

            if (meanAUC > bestAUC) {
                bestAUC = meanAUC;
                bestStd = stdAUC;
                bestDepth = depth;
            }
        }


        // --------------------------------------------------------
        // Compact tuning summary
        // --------------------------------------------------------

        std::cout << "\n========================================"
                  << "\nBDT TUNING SUMMARY"
                  << "\n========================================"
                  << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {

            std::cout << "MaxDepth = "
                      << depthValues[i]
                      << "   Mean AUC = "
                      << results[i].GetROCAverage()
                      << "   Std = "
                      << results[i].GetROCStandardDeviation()
                      << std::endl;
        }


        std::cout << "\n========================================"
                  << "\nSELECTED BDT CONFIGURATION"
                  << "\n========================================"
                  << "\nMaxDepth    = "
                  << bestDepth
                  << "\nBest CV AUC = "
                  << bestAUC
                  << "\nCV AUC std  = "
                  << bestStd
                  << "\n========================================"
                  << std::endl;
    }


    cvFile->Write();
    cvFile->Close();


    // ============================================================
    // 2. FINAL TRAINING
    // ============================================================

    if (gSystem->AccessPathName("dataset_BDT_nj1"))
        gSystem->mkdir("dataset_BDT_nj1");

    if (gSystem->AccessPathName("dataset_BDT_nj1")) {
        std::cerr
            << "ERROR creating dataset_BDT_nj1 directory."
            << std::endl;
        return;
    }

    TString outputPath =
        "dataset_BDT_nj1/BDT_results.root";

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
            "BDT",
            outputFile,
            "!V:!Silent:Color:DrawProgressBar:"
            "Transformations=I:"
            "AnalysisType=Classification");

    TMVA::DataLoader* finalLoader =
        new TMVA::DataLoader("dataset_BDT_nj1");


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
    // Final selected BDT
    // ------------------------------------------------------------

    TString finalOptions =
        Form(
            "!H:!V:"
            "NTrees=800:"
            "MinNodeSize=2.5%%:"
            "MaxDepth=%d:"
            "BoostType=AdaBoost:"
            "AdaBoostBeta=0.5:"
            "UseBaggedBoost:"
            "BaggedSampleFraction=0.5:"
            "SeparationType=GiniIndex:"
            "nCuts=20:"
            "NegWeightTreatment=IgnoreNegWeightsInTraining",
            bestDepth);


    factory->BookMethod(
        finalLoader,
        TMVA::Types::kBDT,
        "BDT",
        finalOptions);


    std::cout << "\n========================================"
              << "\nFINAL BDT TRAINING -- nj = 1"
              << "\nSelected MaxDepth = "
              << bestDepth
              << "\n========================================"
              << std::endl;


    // ============================================================
    // Train / test / evaluate
    // ============================================================

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();


    // ============================================================
    // Final test AUC
    // ============================================================

    double testAUC =
        factory->GetROCIntegral(
            finalLoader,
            "BDT",
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
    TMVA::TMVAGui("dataset_BDT_nj1/BDT_results.root");

// ============================================================
// 4. FINAL SUMMARY
// ============================================================

std::cout << "\n========================================"
          << "\nBDT FINAL SUMMARY -- nj = 1"
          << "\n========================================"
          << "\nSelected MaxDepth = " << bestDepth
          << "\nNTrees            = 800"
          << "\nCV AUC             = " << bestAUC
          << "\nCV AUC std         = " << bestStd
          << "\nFinal test AUC     = " << testAUC
          << "\n"
          << "\nSignal train       = 42378"
          << "\nSignal test        = 10595"
          << "\nBackground train   = 57632"
          << "\nBackground test    = 14408"
          << "\n"
          << "\nOutput ROOT file   = dataset_BDT_nj1/BDT_results.root"
          << "\n========================================"
          << std::endl;

delete cvLoader;

sigFile->Close();
bkgFile->Close();

delete sigFile;
delete bkgFile;
}