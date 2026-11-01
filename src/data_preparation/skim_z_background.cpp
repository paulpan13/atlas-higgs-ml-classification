/**
 * @file skim_z_background.cpp
 * @brief Splits the Z->mumu background samples into jet-multiplicity categories.
 *
 * The three Z/gamma* -> mumu background samples are combined into a TChain
 * and divided into the n_j = 0, n_j = 1, and n_j >= 2 event categories.
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

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"

void skim_z_background()
{
    TChain chain("analysis");

    // Z/gamma* -> mumu background samples
    chain.Add("Zbfilter.root");
    chain.Add("ZCfilterBveto.root");
    chain.Add("ZCvetoBveto.root");

    std::cout << "Total input entries: "
              << chain.GetEntries() << std::endl;

    // n_j = 0
    TFile *f0 = TFile::Open("Zskimmed_nj0.root", "RECREATE");
    TTree *t0 = chain.CopyTree("jet_n == 0");
    t0->Write();
    f0->Close();

    // n_j = 1
    TFile *f1 = TFile::Open("Zskimmed_nj1.root", "RECREATE");
    TTree *t1 = chain.CopyTree("jet_n == 1");
    t1->Write();
    f1->Close();

    // n_j >= 2
    TFile *f2 = TFile::Open("Zskimmed_nj2.root", "RECREATE");
    TTree *t2 = chain.CopyTree("jet_n >= 2");
    t2->Write();
    f2->Close();

    // Summary
    Long64_t nTotal = chain.GetEntries();
    Long64_t n0     = chain.GetEntries("jet_n == 0");
    Long64_t n1     = chain.GetEntries("jet_n == 1");
    Long64_t n2     = chain.GetEntries("jet_n >= 2");

    std::cout << "\n=========== SUMMARY ==========="
              << std::endl;

    std::cout << "Total   : " << nTotal << std::endl;
    std::cout << "n_j = 0 : " << n0 << std::endl;
    std::cout << "n_j = 1 : " << n1 << std::endl;
    std::cout << "n_j >= 2: " << n2 << std::endl;

    std::cout << "\nCheck:" << std::endl;
    std::cout << "n0 + n1 + n2 = "
              << n0 + n1 + n2 << std::endl;

    std::cout << "Original      = "
              << nTotal << std::endl;

    if (n0 + n1 + n2 == nTotal)
        std::cout << "CHECK PASSED" << std::endl;
    else
        std::cout << "CHECK FAILED" << std::endl;

    std::cout << "==============================="
              << std::endl;
}