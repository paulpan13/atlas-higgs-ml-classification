/**
 * @file build_nj1_dataset.cpp
 * @brief Builds the signal and background datasets for the n_j = 1 category.
 *
 * Applies the event selection used in the bachelor's thesis, constructs
 * the dimuon and leading-jet kinematic variables, computes normalized
 * Monte Carlo event weights, and writes the selected events to ROOT TTrees.
 *
 * Part of the bachelor's thesis:
 * "Application of Machine Learning Techniques for Event Classification
 *  in ATLAS Open Data"
 *
 * Author: Pavlos Panagiotopoulos
 * National and Kapodistrian University of Athens
 * 2026
 */

#include <TFile.h>
#include <TTree.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <iomanip>


const double mMu = 0.105658;


// ============================================================
// MC normalization
// ============================================================

struct NormInfo
{
    double xsec;
    double kfac;
    double filteff;
    double sumw;
};


std::map<unsigned int, NormInfo> normInfo =
{
    {345097, {28.3,   1.717, 0.000218, 4247621.5}},
    {345098, {0.1227, 1.0,   0.000275, 2239.9658}},
    {345106, {3.782,  1.0,   0.000275, 448400.06}},

    {700323, {2221.3, 1.0, 0.024394, 1.2741689475072e13}},
    {700324, {2221.3, 1.0, 0.13003,  1.46058376118272e14}},
    {700325, {2221.3, 1.0, 0.846393, 1.19632371174605e15}}
};


double GetNormalization(unsigned int channel)
{
    auto it = normInfo.find(channel);

    if (it == normInfo.end())
    {
        std::cerr
            << "ERROR: unknown channelNumber = "
            << channel
            << std::endl;

        return 0.0;
    }

    const NormInfo& n = it->second;

    return
        n.xsec *
        n.kfac *
        n.filteff /
        n.sumw;
}


// ============================================================
// Output variables
// ============================================================

struct OutputVariables
{
    float pT_mumu;
    float Y_mumu;
    float cosTheta;

    float jet1_pt;
    float jet1_eta;
    float dPhi_mumu_j1;

    float m_mumu;

    ULong64_t eventNumber;
    int jet_n;
    unsigned int channelNumber;

    double eventWeight;
};


// ============================================================
// Create output branches
// ============================================================

void CreateBranches(
    TTree* tree,
    OutputVariables& v)
{
    tree->Branch(
        "pT_mumu",
        &v.pT_mumu,
        "pT_mumu/F"
    );

    tree->Branch(
        "Y_mumu",
        &v.Y_mumu,
        "Y_mumu/F"
    );

    tree->Branch(
        "cosTheta",
        &v.cosTheta,
        "cosTheta/F"
    );

    tree->Branch(
        "jet1_pt",
        &v.jet1_pt,
        "jet1_pt/F"
    );

    tree->Branch(
        "jet1_eta",
        &v.jet1_eta,
        "jet1_eta/F"
    );

    tree->Branch(
        "dPhi_mumu_j1",
        &v.dPhi_mumu_j1,
        "dPhi_mumu_j1/F"
    );

    tree->Branch(
        "m_mumu",
        &v.m_mumu,
        "m_mumu/F"
    );

    tree->Branch(
        "eventNumber",
        &v.eventNumber,
        "eventNumber/l"
    );

    tree->Branch(
        "jet_n",
        &v.jet_n,
        "jet_n/I"
    );

    tree->Branch(
        "channelNumber",
        &v.channelNumber,
        "channelNumber/i"
    );

    tree->Branch(
        "eventWeight",
        &v.eventWeight,
        "eventWeight/D"
    );
}


// ============================================================
// Process one input sample
// ============================================================

void ProcessSample(
    const char* filename,
    TTree* outputTree,
    OutputVariables& out)
{
    TFile* file =
        TFile::Open(filename);

    if (!file ||
        file->IsZombie())
    {
        std::cerr
            << "ERROR opening "
            << filename
            << std::endl;

        return;
    }


    TTree* tree =
        (TTree*)file->Get("analysis");

    if (!tree)
    {
        std::cerr
            << "ERROR: tree 'analysis' not found in "
            << filename
            << std::endl;

        file->Close();
        return;
    }


    // ========================================================
    // Input branches
    // ========================================================

    std::vector<float>* lep_pt = nullptr;
    std::vector<float>* lep_eta = nullptr;
    std::vector<float>* lep_phi = nullptr;

    std::vector<int>* lep_type = nullptr;
    std::vector<int>* lep_charge = nullptr;

    std::vector<char>* lep_isLooseID = nullptr;
    std::vector<char>* lep_isLooseIso = nullptr;
    std::vector<char>* lep_isTrigMatched = nullptr;

    std::vector<float>* lep_z0 = nullptr;
    std::vector<float>* lep_d0sig = nullptr;


    std::vector<float>* jet_pt = nullptr;
    std::vector<float>* jet_eta = nullptr;
    std::vector<float>* jet_phi = nullptr;


    int jet_n = 0;

    unsigned int channelNumber = 0;
    ULong64_t eventNumber = 0;

    float mcWeight = 0.0;

    float ScaleFactor_MUON = 0.0;
    float ScaleFactor_LepTRIGGER = 0.0;
    float ScaleFactor_PILEUP = 0.0;


    tree->SetBranchAddress(
        "lep_pt",
        &lep_pt
    );

    tree->SetBranchAddress(
        "lep_eta",
        &lep_eta
    );

    tree->SetBranchAddress(
        "lep_phi",
        &lep_phi
    );

    tree->SetBranchAddress(
        "lep_type",
        &lep_type
    );

    tree->SetBranchAddress(
        "lep_charge",
        &lep_charge
    );

    tree->SetBranchAddress(
        "lep_isLooseID",
        &lep_isLooseID
    );

    tree->SetBranchAddress(
        "lep_isLooseIso",
        &lep_isLooseIso
    );

    tree->SetBranchAddress(
        "lep_isTrigMatched",
        &lep_isTrigMatched
    );

    tree->SetBranchAddress(
        "lep_z0",
        &lep_z0
    );

    tree->SetBranchAddress(
        "lep_d0sig",
        &lep_d0sig
    );


    tree->SetBranchAddress(
        "jet_n",
        &jet_n
    );

    tree->SetBranchAddress(
        "jet_pt",
        &jet_pt
    );

    tree->SetBranchAddress(
        "jet_eta",
        &jet_eta
    );

    tree->SetBranchAddress(
        "jet_phi",
        &jet_phi
    );


    tree->SetBranchAddress(
        "channelNumber",
        &channelNumber
    );

    tree->SetBranchAddress(
        "eventNumber",
        &eventNumber
    );

    tree->SetBranchAddress(
        "mcWeight",
        &mcWeight
    );

    tree->SetBranchAddress(
        "ScaleFactor_MUON",
        &ScaleFactor_MUON
    );

    tree->SetBranchAddress(
        "ScaleFactor_LepTRIGGER",
        &ScaleFactor_LepTRIGGER
    );

    tree->SetBranchAddress(
        "ScaleFactor_PILEUP",
        &ScaleFactor_PILEUP
    );


    // ========================================================
    // Counters
    // ========================================================

    Long64_t selected = 0;

    Long64_t positive = 0;
    Long64_t negative = 0;
    Long64_t zero = 0;
    Long64_t nonFiniteWeight = 0;

    Long64_t badJetVectors = 0;

    double sumWeights = 0.0;

    Long64_t nEntries =
        tree->GetEntries();


    // ========================================================
    // Event loop
    // ========================================================

    for (Long64_t entry = 0;
         entry < nEntries;
         ++entry)
    {
        tree->GetEntry(entry);


        // ====================================================
        // nj = 1
        // ====================================================

        if (jet_n != 1)
            continue;


        // ====================================================
        // Jet-vector safety
        // ====================================================

        if (!jet_pt ||
            !jet_eta ||
            !jet_phi)
        {
            badJetVectors++;
            continue;
        }

        if (jet_pt->size() < 1 ||
            jet_eta->size() < 1 ||
            jet_phi->size() < 1)
        {
            badJetVectors++;
            continue;
        }

        if (jet_pt->size() != jet_eta->size() ||
            jet_pt->size() != jet_phi->size())
        {
            badJetVectors++;
            continue;
        }


        // ====================================================
        // Qualifying muons
        //
        // |type| = 13
        // pT > 10 GeV
        // Loose ID
        // ====================================================

        std::vector<int> goodMuons;


        for (size_t i = 0;
             i < lep_pt->size();
             ++i)
        {
            if (std::abs(
                    lep_type->at(i)
                ) != 13)
            {
                continue;
            }

            if (lep_pt->at(i) <= 10.0)
                continue;

            if (!lep_isLooseID->at(i))
                continue;


            goodMuons.push_back(
                static_cast<int>(i)
            );
        }


        if (goodMuons.size() < 2)
            continue;


        // ====================================================
        // Highest-pT opposite-sign pair
        // ====================================================

        int idx1 = -1;
        int idx2 = -1;

        double bestLeadPt = -1.0;
        double bestSubPt = -1.0;


        for (size_t a = 0;
             a < goodMuons.size();
             ++a)
        {
            for (size_t b = a + 1;
                 b < goodMuons.size();
                 ++b)
            {
                int i =
                    goodMuons[a];

                int j =
                    goodMuons[b];


                if (lep_charge->at(i) *
                    lep_charge->at(j) >= 0)
                {
                    continue;
                }


                double pti =
                    lep_pt->at(i);

                double ptj =
                    lep_pt->at(j);


                double leadPt =
                    std::max(
                        pti,
                        ptj
                    );

                double subPt =
                    std::min(
                        pti,
                        ptj
                    );


                if (leadPt > bestLeadPt ||
                    (
                        leadPt == bestLeadPt &&
                        subPt > bestSubPt
                    ))
                {
                    bestLeadPt =
                        leadPt;

                    bestSubPt =
                        subPt;

                    idx1 = i;
                    idx2 = j;
                }
            }
        }


        if (idx1 < 0 ||
            idx2 < 0)
        {
            continue;
        }


        // ====================================================
        // Leading / subleading pT
        // ====================================================

        int lead =
            (
                lep_pt->at(idx1) >=
                lep_pt->at(idx2)
            )
            ? idx1
            : idx2;


        int sub =
            (lead == idx1)
            ? idx2
            : idx1;


        if (lep_pt->at(lead) <= 27.0)
            continue;

        if (lep_pt->at(sub) <= 15.0)
            continue;


        // ====================================================
        // eta
        // ====================================================

        if (std::abs(
                lep_eta->at(idx1)
            ) >= 2.5)
        {
            continue;
        }


        if (std::abs(
                lep_eta->at(idx2)
            ) >= 2.5)
        {
            continue;
        }


        // ====================================================
        // Impact parameters
        // ====================================================

        double z01 =
            lep_z0->at(idx1) /
            std::cosh(
                lep_eta->at(idx1)
            );


        double z02 =
            lep_z0->at(idx2) /
            std::cosh(
                lep_eta->at(idx2)
            );


        if (std::abs(z01) >= 0.5)
            continue;

        if (std::abs(z02) >= 0.5)
            continue;


        if (std::abs(
                lep_d0sig->at(idx1)
            ) >= 3.0)
        {
            continue;
        }


        if (std::abs(
                lep_d0sig->at(idx2)
            ) >= 3.0)
        {
            continue;
        }


        // ====================================================
        // Isolation
        // ====================================================

        if (!lep_isLooseIso->at(idx1))
            continue;

        if (!lep_isLooseIso->at(idx2))
            continue;


        // ====================================================
        // Trigger
        // ====================================================

        if (!lep_isTrigMatched->at(idx1) &&
            !lep_isTrigMatched->at(idx2))
        {
            continue;
        }


        // ====================================================
        // Additional-lepton veto
        //
        // Additional stored e/mu:
        // pT > 7 GeV
        // Loose ID
        // Candidate muons excluded.
        // ====================================================

        bool extraLepton = false;


        for (size_t i = 0;
             i < lep_pt->size();
             ++i)
        {
            if (static_cast<int>(i) == idx1 ||
                static_cast<int>(i) == idx2)
            {
                continue;
            }


            if (std::abs(
                    lep_type->at(i)
                ) != 11 &&
                std::abs(
                    lep_type->at(i)
                ) != 13)
            {
                continue;
            }


            if (lep_pt->at(i) <= 7.0)
                continue;


            if (!lep_isLooseID->at(i))
                continue;


            extraLepton = true;
            break;
        }


        if (extraLepton)
            continue;


        // ====================================================
        // Muon four-vectors
        // ====================================================

        double pt1 =
            lep_pt->at(idx1);

        double eta1 =
            lep_eta->at(idx1);

        double phi1 =
            lep_phi->at(idx1);


        double pt2 =
            lep_pt->at(idx2);

        double eta2 =
            lep_eta->at(idx2);

        double phi2 =
            lep_phi->at(idx2);


        double px1 =
            pt1 *
            std::cos(phi1);

        double py1 =
            pt1 *
            std::sin(phi1);

        double pz1 =
            pt1 *
            std::sinh(eta1);


        double px2 =
            pt2 *
            std::cos(phi2);

        double py2 =
            pt2 *
            std::sin(phi2);

        double pz2 =
            pt2 *
            std::sinh(eta2);


        double E1 =
            std::sqrt(
                pt1 * pt1 *
                std::cosh(eta1) *
                std::cosh(eta1)
                +
                mMu * mMu
            );


        double E2 =
            std::sqrt(
                pt2 * pt2 *
                std::cosh(eta2) *
                std::cosh(eta2)
                +
                mMu * mMu
            );


        // ====================================================
        // Dimuon four-vector
        // ====================================================

        double px =
            px1 + px2;

        double py =
            py1 + py2;

        double pz =
            pz1 + pz2;

        double E =
            E1 + E2;


        // ====================================================
        // Invariant mass
        // ====================================================

        double mass2 =
            E * E
            -
            px * px
            -
            py * py
            -
            pz * pz;


        if (mass2 <= 0.0)
            continue;


        double m_mumu =
            std::sqrt(mass2);


        if (m_mumu <= 120.0 ||
            m_mumu >= 130.0)
        {
            continue;
        }


        // ====================================================
        // INPUT 1: pT_mumu
        // ====================================================

        double pT_mumu =
            std::sqrt(
                px * px +
                py * py
            );


        // ====================================================
        // INPUT 2: exact dimuon rapidity
        // ====================================================

        double rapidityNumerator =
            E + pz;

        double rapidityDenominator =
            E - pz;


        if (rapidityNumerator <= 0.0 ||
            rapidityDenominator <= 0.0)
        {
            continue;
        }


        double Y_mumu =
            0.5 *
            std::log(
                rapidityNumerator /
                rapidityDenominator
            );


        // ====================================================
        // INPUT 3: Collins-Soper cos(theta*)
        //
        // particle 1 = mu-
        // particle 2 = mu+
        // ====================================================

        int iMinus =
            (lep_charge->at(idx1) < 0)
            ? idx1
            : idx2;


        int iPlus =
            (lep_charge->at(idx1) < 0)
            ? idx2
            : idx1;


        double ptMinus =
            lep_pt->at(iMinus);

        double etaMinus =
            lep_eta->at(iMinus);


        double ptPlus =
            lep_pt->at(iPlus);

        double etaPlus =
            lep_eta->at(iPlus);


        double pzMinus =
            ptMinus *
            std::sinh(etaMinus);

        double pzPlus =
            ptPlus *
            std::sinh(etaPlus);


        double EMinus =
            std::sqrt(
                ptMinus * ptMinus *
                std::cosh(etaMinus) *
                std::cosh(etaMinus)
                +
                mMu * mMu
            );


        double EPlus =
            std::sqrt(
                ptPlus * ptPlus *
                std::cosh(etaPlus) *
                std::cosh(etaPlus)
                +
                mMu * mMu
            );


        double P1plus =
            (EMinus + pzMinus) /
            std::sqrt(2.0);

        double P1minus =
            (EMinus - pzMinus) /
            std::sqrt(2.0);

        double P2plus =
            (EPlus + pzPlus) /
            std::sqrt(2.0);

        double P2minus =
            (EPlus - pzPlus) /
            std::sqrt(2.0);


        double denominatorCS =
            std::sqrt(
                m_mumu *
                m_mumu *
                (
                    m_mumu *
                    m_mumu
                    +
                    pT_mumu *
                    pT_mumu
                )
            );


        if (denominatorCS <= 0.0)
            continue;


        if (pz == 0.0)
            continue;


        double signPz =
            (pz > 0.0)
            ? 1.0
            : -1.0;


        double cosTheta =
            2.0 *
            (
                P1plus *
                P2minus
                -
                P1minus *
                P2plus
            )
            /
            denominatorCS
            *
            signPz;


        // ====================================================
        // INPUT 4: jet1_pt
        //
        // Since jet_n == 1, element [0] is the unique
        // stored jet in the event.
        // ====================================================

        double jet1_pt =
            jet_pt->at(0);


        // ====================================================
        // INPUT 5: jet1_eta
        // ====================================================

        double jet1_eta =
            jet_eta->at(0);


        // ====================================================
        // INPUT 6: DeltaPhi(mumu, j1)
        // ====================================================

        double jet1_phi =
            jet_phi->at(0);


        double phi_mumu =
            std::atan2(
                py,
                px
            );


        double rawDPhi =
            phi_mumu -
            jet1_phi;


        double wrappedDPhi =
            std::atan2(
                std::sin(rawDPhi),
                std::cos(rawDPhi)
            );


        double dPhi_mumu_j1 =
            std::abs(
                wrappedDPhi
            );


        // ====================================================
        // Numerical sanity
        // ====================================================

        if (!std::isfinite(pT_mumu) ||
            !std::isfinite(Y_mumu) ||
            !std::isfinite(cosTheta) ||
            !std::isfinite(jet1_pt) ||
            !std::isfinite(jet1_eta) ||
            !std::isfinite(dPhi_mumu_j1) ||
            !std::isfinite(m_mumu))
        {
            continue;
        }


        // ====================================================
        // Physical normalized event weight
        // ====================================================

        double eventWeight =
            static_cast<double>(mcWeight)
            *
            static_cast<double>(ScaleFactor_MUON)
            *
            static_cast<double>(ScaleFactor_LepTRIGGER)
            *
            static_cast<double>(ScaleFactor_PILEUP)
            *
            GetNormalization(channelNumber);


        if (!std::isfinite(eventWeight))
        {
            nonFiniteWeight++;
            continue;
        }


        // ====================================================
        // Fill output tree
        // ====================================================

        out.pT_mumu =
            static_cast<float>(
                pT_mumu
            );


        out.Y_mumu =
            static_cast<float>(
                Y_mumu
            );


        out.cosTheta =
            static_cast<float>(
                cosTheta
            );


        out.jet1_pt =
            static_cast<float>(
                jet1_pt
            );


        out.jet1_eta =
            static_cast<float>(
                jet1_eta
            );


        out.dPhi_mumu_j1 =
            static_cast<float>(
                dPhi_mumu_j1
            );


        out.m_mumu =
            static_cast<float>(
                m_mumu
            );


        out.eventNumber =
            eventNumber;


        out.jet_n =
            jet_n;


        out.channelNumber =
            channelNumber;


        out.eventWeight =
            eventWeight;


        outputTree->Fill();


        // ====================================================
        // Audit counters
        // ====================================================

        selected++;

        sumWeights +=
            eventWeight;


        if (eventWeight > 0.0)
            positive++;

        else if (eventWeight < 0.0)
            negative++;

        else
            zero++;
    }


    // ========================================================
    // Sample summary
    // ========================================================

    std::cout
        << "\n"
        << filename
        << "\n";


    std::cout
        << "  selected          = "
        << selected
        << "\n";


    std::cout
        << "  positive          = "
        << positive
        << "\n";


    std::cout
        << "  negative          = "
        << negative
        << "\n";


    std::cout
        << "  zero              = "
        << zero
        << "\n";


    std::cout
        << "  non-finite weight = "
        << nonFiniteWeight
        << "\n";


    std::cout
        << "  bad jet vectors   = "
        << badJetVectors
        << "\n";


    std::cout
        << "  sum(eventWeight)  = "
        << std::setprecision(12)
        << sumWeights
        << "\n";


    file->Close();
}


// ============================================================
// Main
// ============================================================

void build_nj1_dataset()
{
    // ========================================================
    // SIGNAL OUTPUT
    // ========================================================

    TFile* signalFile =
        new TFile(
            "nj1_signal.root",
            "RECREATE"
        );


    TTree* signalTree =
        new TTree(
            "signal",
            "Final nj=1 signal dataset"
        );


    OutputVariables signalOut;


    CreateBranches(
        signalTree,
        signalOut
    );


    std::cout
        << "SIGNAL\n";


    ProcessSample(
        "ggH.root",
        signalTree,
        signalOut
    );


    ProcessSample(
        "ggZH.root",
        signalTree,
        signalOut
    );


    ProcessSample(
        "VBF.root",
        signalTree,
        signalOut
    );


    Long64_t finalSignalEntries =
        signalTree->GetEntries();


    signalFile->cd();

    signalTree->Write();

    signalFile->Close();


    // ========================================================
    // BACKGROUND OUTPUT
    // ========================================================

    TFile* backgroundFile =
        new TFile(
            "nj1_background.root",
            "RECREATE"
        );


    TTree* backgroundTree =
        new TTree(
            "background",
            "Final nj=1 background dataset"
        );


    OutputVariables backgroundOut;


    CreateBranches(
        backgroundTree,
        backgroundOut
    );


    std::cout
        << "BACKGROUND\n";


    ProcessSample(
        "Zskimmed_nj1.root",
        backgroundTree,
        backgroundOut
    );


    Long64_t finalBackgroundEntries =
        backgroundTree->GetEntries();


    backgroundFile->cd();

    backgroundTree->Write();

    backgroundFile->Close();


    // ========================================================
    // Final summary
    // ========================================================

    std::cout
        << "\n========================================\n"
        << "        FINAL nj=1 DATASET SUMMARY\n"
        << "========================================\n"
        << "Signal entries     = "
        << finalSignalEntries
        << "\n"
        << "Background entries = "
        << finalBackgroundEntries
        << "\n"
        << "========================================\n"
        << "\nCreated:\n"
        << "  nj1_signal.root\n"
        << "  nj1_background.root\n"
        << std::endl;
}