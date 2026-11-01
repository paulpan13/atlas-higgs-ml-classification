/**
 * @file likelihood.cpp
 * @brief Tunes, trains, and evaluates a Projective Likelihood classifier
 *        for the n_j = 0 category.
 *
 * Performs 5-fold cross-validation to optimize the number of probability
 * density estimation bins (Nbins), then trains the final classifier using
 * the selected configuration and evaluates it on the fixed test sample.
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

void likelihood()
{
    TMVA::Tools::Instance();

    std::vector<int> nBinsValues = {50, 100, 150, 200, 250, 300};

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
    // 1. CROSS-VALIDATION
    // ============================================================

    TFile* cvFile = TFile::Open("Likelihood_CV_results.root", "RECREATE");

    if (!cvFile || cvFile->IsZombie()) {
        std::cerr << "ERROR creating CV output file." << std::endl;
        sigFile->Close();
        bkgFile->Close();
        return;
    }

    TMVA::DataLoader* cvLoader = new TMVA::DataLoader("cv_Likelihood");

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

    TString cvOptions =
        "!V:"
        "!Silent:"
        "!ModelPersistence:"
        "AnalysisType=Classification:"
        "SplitType=RandomStratified:"
        "NumFolds=5";

    TMVA::CrossValidation cv(
        "Likelihood_CV",
        cvLoader,
        cvFile,
        cvOptions);

    for (int nBins : nBinsValues) {
        TString methodName = Form("Likelihood_Nbins%d", nBins);

        TString methodOptions = Form(
            "!H:!V:"
            "TransformOutput=false:"
            "PDFInterpol=Spline2:"
            "NSmooth=1:"
            "NAvEvtPerBin=0:"
            "Nbins=%d:"
            "VarTransform=None:"
            "IgnoreNegWeightsInTraining",
            nBins);

        cv.BookMethod(
            TMVA::Types::kLikelihood,
            methodName,
            methodOptions);
    }

    std::cout << "\n========================================"
              << "\nLIKELIHOOD HYPERPARAMETER TUNING"
              << "\n5-fold cross-validation"
              << "\n========================================"
              << std::endl;

    cv.Evaluate();

    const auto& results = cv.GetResults();

    if (results.size() != nBinsValues.size()) {
        std::cerr << "\nERROR: unexpected number of CV results."
                  << "\nExpected: " << nBinsValues.size()
                  << "\nReceived: " << results.size()
                  << std::endl;
        return;
    }

    double bestAUC = -1.0;
    double bestStd = 0.0;
    int bestNBins = -1;

    std::cout << "\n========================================"
              << "\nLIKELIHOOD CROSS-VALIDATION RESULTS"
              << "\n========================================"
              << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        int nBins = nBinsValues[i];

        double meanAUC = results[i].GetROCAverage();
        double stdAUC = results[i].GetROCStandardDeviation();
        const auto& foldAUCs = results[i].GetROCValues();

        std::cout << "\nNbins = " << nBins
                  << "\nFold AUCs:" << std::endl;

        for (const auto& entry : foldAUCs)
            std::cout << " Fold " << entry.first + 1
                      << ": " << entry.second
                      << std::endl;

        std::cout << "Mean CV AUC = " << meanAUC
                  << "\nCV AUC std  = " << stdAUC
                  << std::endl;

        if (meanAUC > bestAUC) {
            bestAUC = meanAUC;
            bestStd = stdAUC;
            bestNBins = nBins;
        }
    }

    std::cout << "\n========================================"
              << "\nLIKELIHOOD TUNING SUMMARY"
              << "\n========================================"
              << std::endl;

    for (size_t i = 0; i < results.size(); ++i)
        std::cout << "Nbins = " << nBinsValues[i]
                  << "   Mean AUC = " << results[i].GetROCAverage()
                  << "   Std = " << results[i].GetROCStandardDeviation()
                  << std::endl;

    std::cout << "\nSelected Nbins = " << bestNBins
              << "\nBest CV AUC    = " << bestAUC
              << "\nCV AUC std     = " << bestStd
              << "\n========================================"
              << std::endl;

    cvFile->Write();
    cvFile->Close();

    // ============================================================
    // 2. FINAL TRAINING
    // ============================================================

    if (gSystem->AccessPathName("dataset_Likelihood"))
        gSystem->mkdir("dataset_Likelihood");

    if (gSystem->AccessPathName("dataset_Likelihood")) {
        std::cerr << "ERROR creating dataset_Likelihood directory." << std::endl;
        return;
    }

    TString outputPath = "dataset_Likelihood/Likelihood_results.root";
    TFile* outputFile = TFile::Open(outputPath, "RECREATE");

    if (!outputFile || outputFile->IsZombie()) {
        std::cerr << "ERROR creating final output file." << std::endl;
        return;
    }

    TMVA::Factory* factory = new TMVA::Factory(
        "Likelihood",
        outputFile,
        "!V:!Silent:Color:DrawProgressBar:"
        "Transformations=I:"
        "AnalysisType=Classification");

    TMVA::DataLoader* finalLoader =
        new TMVA::DataLoader("dataset_Likelihood");

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

    TString finalOptions = Form(
        "!H:!V:"
        "TransformOutput=false:"
        "PDFInterpol=Spline2:"
        "NSmooth=1:"
        "NAvEvtPerBin=0:"
        "Nbins=%d:"
        "VarTransform=None:"
        "IgnoreNegWeightsInTraining",
        bestNBins);

    factory->BookMethod(
        finalLoader,
        TMVA::Types::kLikelihood,
        "Likelihood",
        finalOptions);

    std::cout << "\n========================================"
              << "\nFINAL LIKELIHOOD TRAINING"
              << "\nSelected Nbins = " << bestNBins
              << "\n========================================"
              << std::endl;

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    double testAUC = factory->GetROCIntegral(
        finalLoader,
        "Likelihood",
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
    TMVA::TMVAGui("dataset_Likelihood/Likelihood_results.root");

// ============================================================
// 4. FINAL SUMMARY
// ============================================================

std::cout << "\n========================================"
          << "\nLIKELIHOOD FINAL SUMMARY"
          << "\n========================================"
          << "\nSelected Nbins    = " << bestNBins
          << "\nCV AUC            = " << bestAUC
          << "\nCV AUC std        = " << bestStd
          << "\nFinal test AUC    = " << testAUC
          << "\n"
          << "\nSignal train      = 46273"
          << "\nSignal test       = 11569"
          << "\nBackground train  = 31770"
          << "\nBackground test   = 7943"
          << "\n"
          << "\nOutput ROOT file  = dataset_Likelihood/Likelihood_results.root"
          << "\n========================================"
          << std::endl;

delete cvLoader;

sigFile->Close();
bkgFile->Close();

delete sigFile;
delete bkgFile;
}