/**
 * @file build_nj2plus_dataset.cpp
 * @brief Builds the signal and background datasets for the n_j >= 2 category.
 *
 * Applies the event selection used in the bachelor's thesis, constructs
 * dimuon, jet, dijet, missing-transverse-momentum, and event-level
 * kinematic variables, computes normalized Monte Carlo event weights,
 * and writes the selected events to ROOT TTrees.
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
#include <algorithm>
#include <cstdint>

const double mMu = 0.105658;


// ============================================================
// MC normalization
// ============================================================

double GetNormalization(unsigned int channelNumber)
{
    double xsec = 0.0;
    double kfac = 1.0;
    double filteff = 0.0;
    double sumw = 0.0;

    switch (channelNumber)
    {
        case 345097: // ggH
            xsec = 28.3;
            kfac = 1.717;
            filteff = 0.000218;
            sumw = 4247621.5;
            break;

        case 345098: // ggZH
            xsec = 0.1227;
            kfac = 1.0;
            filteff = 0.000275;
            sumw = 2239.9658;
            break;

        case 345106: // VBF
            xsec = 3.782;
            kfac = 1.0;
            filteff = 0.000275;
            sumw = 448400.06;
            break;

        case 700323: // Z BFilter
            xsec = 2221.3;
            kfac = 1.0;
            filteff = 0.024394;
            sumw = 1.2741689475072e13;
            break;

        case 700324: // Z CFilterBVeto
            xsec = 2221.3;
            kfac = 1.0;
            filteff = 0.13003;
            sumw = 1.46058376118272e14;
            break;

        case 700325: // Z CVetoBVeto
            xsec = 2221.3;
            kfac = 1.0;
            filteff = 0.846393;
            sumw = 1.19632371174605e15;
            break;

        default:
            std::cerr << "Unknown channelNumber: "
                      << channelNumber << std::endl;
            return 0.0;
    }

    return xsec * kfac * filteff / sumw;
}


// ============================================================
// Delta phi in [0, pi]
// ============================================================

double DeltaPhi(double phi1, double phi2)
{
    double dphi = phi1 - phi2;
    return std::abs(std::atan2(std::sin(dphi), std::cos(dphi)));
}


// ============================================================
// Process one input file
// ============================================================

void ProcessFile(const char* inputFile,
                 TTree* outputTree,
                 Long64_t& nSelected,
                 Long64_t& nPositive,
                 Long64_t& nNegative,
                 Long64_t& nZero,
                 Long64_t& nNonFinite,
                 Long64_t& nBadJetVectors,
                 double& sumEventWeight,

                 float& out_pT_mumu,
                 float& out_Y_mumu,
                 float& out_cosTheta,

                 float& out_jet1_pt,
                 float& out_jet1_eta,
                 float& out_dPhi_j1_mumu,

                 float& out_jet2_pt,
                 float& out_jet2_eta,
                 float& out_dPhi_j2_mumu,

                 float& out_pT_jj,
                 float& out_Y_jj,
                 float& out_dPhi_jj_mumu,

                 float& out_m_jj,
                 float& out_met,
                 float& out_HT,

                 float& out_m_mumu,
                 ULong64_t& out_eventNumber,
                 int& out_jet_n,
                 unsigned int& out_channelNumber,
                 double& out_eventWeight)
{
    TFile* file = TFile::Open(inputFile, "READ");

    if (!file || file->IsZombie())
    {
        std::cerr << "Cannot open " << inputFile << std::endl;
        return;
    }

    TTree* tree = (TTree*)file->Get("analysis");

    if (!tree)
    {
        std::cerr << "Tree 'analysis' not found in "
                  << inputFile << std::endl;
        file->Close();
        return;
    }


    // --------------------------------------------------------
    // Input branches
    // --------------------------------------------------------

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
    std::vector<float>* jet_e = nullptr;

    int jet_n = 0;

    float met = 0.0;

    ULong64_t eventNumber = 0;
    unsigned int channelNumber = 0;

    float mcWeight = 0.0;
    float ScaleFactor_MUON = 0.0;
    float ScaleFactor_LepTRIGGER = 0.0;
    float ScaleFactor_PILEUP = 0.0;


    tree->SetBranchAddress("lep_pt", &lep_pt);
    tree->SetBranchAddress("lep_eta", &lep_eta);
    tree->SetBranchAddress("lep_phi", &lep_phi);
    tree->SetBranchAddress("lep_type", &lep_type);
    tree->SetBranchAddress("lep_charge", &lep_charge);

    tree->SetBranchAddress("lep_isLooseID", &lep_isLooseID);
    tree->SetBranchAddress("lep_isLooseIso", &lep_isLooseIso);
    tree->SetBranchAddress("lep_isTrigMatched", &lep_isTrigMatched);

    tree->SetBranchAddress("lep_z0", &lep_z0);
    tree->SetBranchAddress("lep_d0sig", &lep_d0sig);

    tree->SetBranchAddress("jet_n", &jet_n);
    tree->SetBranchAddress("jet_pt", &jet_pt);
    tree->SetBranchAddress("jet_eta", &jet_eta);
    tree->SetBranchAddress("jet_phi", &jet_phi);
    tree->SetBranchAddress("jet_e", &jet_e);

    tree->SetBranchAddress("met", &met);

    tree->SetBranchAddress("eventNumber", &eventNumber);
    tree->SetBranchAddress("channelNumber", &channelNumber);

    tree->SetBranchAddress("mcWeight", &mcWeight);
    tree->SetBranchAddress("ScaleFactor_MUON", &ScaleFactor_MUON);
    tree->SetBranchAddress("ScaleFactor_LepTRIGGER", &ScaleFactor_LepTRIGGER);
    tree->SetBranchAddress("ScaleFactor_PILEUP", &ScaleFactor_PILEUP);


    // --------------------------------------------------------
    // Event loop
    // --------------------------------------------------------

    Long64_t nEntries = tree->GetEntries();

    for (Long64_t entry = 0; entry < nEntries; ++entry)
    {
        tree->GetEntry(entry);

        if (jet_n < 2) continue;

        if (!jet_pt || !jet_eta || !jet_phi || !jet_e)
        {
            nBadJetVectors++;
            continue;
        }

        if (jet_pt->size() < 2 ||
            jet_pt->size() != jet_eta->size() ||
            jet_pt->size() != jet_phi->size() ||
            jet_pt->size() != jet_e->size())
        {
            nBadJetVectors++;
            continue;
        }


        // ----------------------------------------------------
        // Find the two highest-pT jets
        // ----------------------------------------------------

        int j1 = -1;
        int j2 = -1;

        for (size_t j = 0; j < jet_pt->size(); ++j)
        {
            if (j1 < 0 || jet_pt->at(j) > jet_pt->at(j1))
            {
                j2 = j1;
                j1 = static_cast<int>(j);
            }
            else if (j2 < 0 || jet_pt->at(j) > jet_pt->at(j2))
            {
                j2 = static_cast<int>(j);
            }
        }

        if (j1 < 0 || j2 < 0)
        {
            nBadJetVectors++;
            continue;
        }


        // ----------------------------------------------------
        // Qualifying muons
        // ----------------------------------------------------

        std::vector<int> goodMuons;

        for (size_t i = 0; i < lep_pt->size(); ++i)
        {
            if (std::abs(lep_type->at(i)) != 13) continue;
            if (lep_pt->at(i) <= 10.0) continue;
            if (!lep_isLooseID->at(i)) continue;

            goodMuons.push_back(static_cast<int>(i));
        }

        if (goodMuons.size() < 2) continue;


        // ----------------------------------------------------
        // Highest-pT opposite-sign dimuon candidate
        // ----------------------------------------------------

        int idx1 = -1;
        int idx2 = -1;

        double bestLeadPt = -1.0;
        double bestSubPt = -1.0;

        for (size_t a = 0; a < goodMuons.size(); ++a)
        {
            for (size_t b = a + 1; b < goodMuons.size(); ++b)
            {
                int i = goodMuons[a];
                int j = goodMuons[b];

                if (lep_charge->at(i) * lep_charge->at(j) >= 0) continue;

                double pti = lep_pt->at(i);
                double ptj = lep_pt->at(j);

                double leadPt = std::max(pti, ptj);
                double subPt = std::min(pti, ptj);

                if (leadPt > bestLeadPt ||
                    (leadPt == bestLeadPt && subPt > bestSubPt))
                {
                    bestLeadPt = leadPt;
                    bestSubPt = subPt;

                    idx1 = i;
                    idx2 = j;
                }
            }
        }

        if (idx1 < 0 || idx2 < 0) continue;

        int lead = (lep_pt->at(idx1) >= lep_pt->at(idx2)) ? idx1 : idx2;
        int sub = (lead == idx1) ? idx2 : idx1;


        // ----------------------------------------------------
        // Final event selection
        // ----------------------------------------------------

        if (lep_pt->at(lead) <= 27.0 ||
            lep_pt->at(sub) <= 15.0) continue;

        if (std::abs(lep_eta->at(idx1)) >= 2.5 ||
            std::abs(lep_eta->at(idx2)) >= 2.5) continue;

        double z01 = lep_z0->at(idx1) / std::cosh(lep_eta->at(idx1));
        double z02 = lep_z0->at(idx2) / std::cosh(lep_eta->at(idx2));

        if (std::abs(z01) >= 0.5 ||
            std::abs(z02) >= 0.5) continue;

        if (std::abs(lep_d0sig->at(idx1)) >= 3.0 ||
            std::abs(lep_d0sig->at(idx2)) >= 3.0) continue;

        if (!lep_isLooseIso->at(idx1) ||
            !lep_isLooseIso->at(idx2)) continue;

        if (!lep_isTrigMatched->at(idx1) &&
            !lep_isTrigMatched->at(idx2)) continue;


        // ----------------------------------------------------
        // Additional-lepton veto
        // ----------------------------------------------------

        bool extraLepton = false;

        for (size_t i = 0; i < lep_pt->size(); ++i)
        {
            if (static_cast<int>(i) == idx1 ||
                static_cast<int>(i) == idx2) continue;

            int absType = std::abs(lep_type->at(i));

            if (absType != 11 && absType != 13) continue;
            if (lep_pt->at(i) <= 7.0) continue;
            if (!lep_isLooseID->at(i)) continue;

            extraLepton = true;
            break;
        }

        if (extraLepton) continue;


        // ----------------------------------------------------
        // Dimuon four-vector
        // ----------------------------------------------------

        double pt1 = lep_pt->at(idx1);
        double eta1 = lep_eta->at(idx1);
        double phi1 = lep_phi->at(idx1);

        double pt2 = lep_pt->at(idx2);
        double eta2 = lep_eta->at(idx2);
        double phi2 = lep_phi->at(idx2);

        double px1 = pt1 * std::cos(phi1);
        double py1 = pt1 * std::sin(phi1);
        double pz1 = pt1 * std::sinh(eta1);

        double px2 = pt2 * std::cos(phi2);
        double py2 = pt2 * std::sin(phi2);
        double pz2 = pt2 * std::sinh(eta2);

        double E1 = std::sqrt(pt1*pt1*std::cosh(eta1)*std::cosh(eta1) + mMu*mMu);
        double E2 = std::sqrt(pt2*pt2*std::cosh(eta2)*std::cosh(eta2) + mMu*mMu);

        double px_mumu = px1 + px2;
        double py_mumu = py1 + py2;
        double pz_mumu = pz1 + pz2;
        double E_mumu = E1 + E2;

        double mass2 = E_mumu*E_mumu
                     - px_mumu*px_mumu
                     - py_mumu*py_mumu
                     - pz_mumu*pz_mumu;

        if (mass2 <= 0.0) continue;

        double m_mumu = std::sqrt(mass2);

        if (m_mumu <= 120.0 || m_mumu >= 130.0) continue;


        // ----------------------------------------------------
        // Dimuon variables
        // ----------------------------------------------------

        double pT_mumu = std::sqrt(px_mumu*px_mumu + py_mumu*py_mumu);

        double rapidityPlus = E_mumu + pz_mumu;
        double rapidityMinus = E_mumu - pz_mumu;

        if (rapidityPlus <= 0.0 || rapidityMinus <= 0.0) continue;

        double Y_mumu = 0.5 * std::log(rapidityPlus / rapidityMinus);

        double phi_mumu = std::atan2(py_mumu, px_mumu);


        // ----------------------------------------------------
        // Collins-Soper cos(theta*)
        // ----------------------------------------------------

        int iMinus = (lep_charge->at(idx1) < 0) ? idx1 : idx2;
        int iPlus = (lep_charge->at(idx1) < 0) ? idx2 : idx1;

        double ptMinus = lep_pt->at(iMinus);
        double etaMinus = lep_eta->at(iMinus);

        double ptPlus = lep_pt->at(iPlus);
        double etaPlus = lep_eta->at(iPlus);

        double pzMinus = ptMinus * std::sinh(etaMinus);
        double pzPlus = ptPlus * std::sinh(etaPlus);

        double EMinus = std::sqrt(
            ptMinus*ptMinus*std::cosh(etaMinus)*std::cosh(etaMinus) + mMu*mMu);

        double EPlus = std::sqrt(
            ptPlus*ptPlus*std::cosh(etaPlus)*std::cosh(etaPlus) + mMu*mMu);

        double P1plus = (EMinus + pzMinus) / std::sqrt(2.0);
        double P1minus = (EMinus - pzMinus) / std::sqrt(2.0);

        double P2plus = (EPlus + pzPlus) / std::sqrt(2.0);
        double P2minus = (EPlus - pzPlus) / std::sqrt(2.0);

        double denominatorCS = std::sqrt(
            m_mumu*m_mumu *
            (m_mumu*m_mumu + pT_mumu*pT_mumu));

        if (denominatorCS <= 0.0 || !std::isfinite(denominatorCS)) continue;

        double signPz = 0.0;
        if (pz_mumu > 0.0) signPz = 1.0;
        else if (pz_mumu < 0.0) signPz = -1.0;

        double cosTheta =
            2.0 * (P1plus*P2minus - P1minus*P2plus)
            / denominatorCS * signPz;


        // ----------------------------------------------------
        // j1 and j2
        // ----------------------------------------------------

        double jet1_pt = jet_pt->at(j1);
        double jet1_eta = jet_eta->at(j1);
        double jet1_phi = jet_phi->at(j1);
        double jet1_E = jet_e->at(j1);

        double jet2_pt = jet_pt->at(j2);
        double jet2_eta = jet_eta->at(j2);
        double jet2_phi = jet_phi->at(j2);
        double jet2_E = jet_e->at(j2);

        double dPhi_j1_mumu = DeltaPhi(jet1_phi, phi_mumu);
        double dPhi_j2_mumu = DeltaPhi(jet2_phi, phi_mumu);


        // ----------------------------------------------------
        // Dijet four-vector
        // ----------------------------------------------------

        double j1_px = jet1_pt * std::cos(jet1_phi);
        double j1_py = jet1_pt * std::sin(jet1_phi);
        double j1_pz = jet1_pt * std::sinh(jet1_eta);

        double j2_px = jet2_pt * std::cos(jet2_phi);
        double j2_py = jet2_pt * std::sin(jet2_phi);
        double j2_pz = jet2_pt * std::sinh(jet2_eta);

        double jj_px = j1_px + j2_px;
        double jj_py = j1_py + j2_py;
        double jj_pz = j1_pz + j2_pz;
        double jj_E = jet1_E + jet2_E;

        double pT_jj = std::sqrt(jj_px*jj_px + jj_py*jj_py);

        double jjRapidityPlus = jj_E + jj_pz;
        double jjRapidityMinus = jj_E - jj_pz;

        if (jjRapidityPlus <= 0.0 ||
            jjRapidityMinus <= 0.0) continue;

        double Y_jj =
            0.5 * std::log(jjRapidityPlus / jjRapidityMinus);

        double phi_jj = std::atan2(jj_py, jj_px);
        double dPhi_jj_mumu = DeltaPhi(phi_jj, phi_mumu);

        double m2_jj =
            jj_E*jj_E
            - jj_px*jj_px
            - jj_py*jj_py
            - jj_pz*jj_pz;

        if (m2_jj < 0.0 && m2_jj > -1e-6) m2_jj = 0.0;
        if (m2_jj < 0.0) continue;

        double m_jj = std::sqrt(m2_jj);


        // ----------------------------------------------------
        // HT = scalar sum of pT of all stored jets
        // ----------------------------------------------------

        double HT = 0.0;

        for (size_t j = 0; j < jet_pt->size(); ++j)
            HT += jet_pt->at(j);


        // ----------------------------------------------------
        // Event weight
        // ----------------------------------------------------

        double correctedWeight =
            static_cast<double>(mcWeight) *
            static_cast<double>(ScaleFactor_MUON) *
            static_cast<double>(ScaleFactor_LepTRIGGER) *
            static_cast<double>(ScaleFactor_PILEUP);

        double normalization = GetNormalization(channelNumber);

        double eventWeight = correctedWeight * normalization;


        // ----------------------------------------------------
        // Final numerical safety check
        // ----------------------------------------------------

        bool finite =
            std::isfinite(pT_mumu) &&
            std::isfinite(Y_mumu) &&
            std::isfinite(cosTheta) &&
            std::isfinite(jet1_pt) &&
            std::isfinite(jet1_eta) &&
            std::isfinite(dPhi_j1_mumu) &&
            std::isfinite(jet2_pt) &&
            std::isfinite(jet2_eta) &&
            std::isfinite(dPhi_j2_mumu) &&
            std::isfinite(pT_jj) &&
            std::isfinite(Y_jj) &&
            std::isfinite(dPhi_jj_mumu) &&
            std::isfinite(m_jj) &&
            std::isfinite(met) &&
            std::isfinite(HT) &&
            std::isfinite(m_mumu) &&
            std::isfinite(eventWeight);

        if (!finite)
        {
            nNonFinite++;
            continue;
        }


        // ----------------------------------------------------
        // Fill output
        // ----------------------------------------------------

        out_pT_mumu = static_cast<float>(pT_mumu);
        out_Y_mumu = static_cast<float>(Y_mumu);
        out_cosTheta = static_cast<float>(cosTheta);

        out_jet1_pt = static_cast<float>(jet1_pt);
        out_jet1_eta = static_cast<float>(jet1_eta);
        out_dPhi_j1_mumu = static_cast<float>(dPhi_j1_mumu);

        out_jet2_pt = static_cast<float>(jet2_pt);
        out_jet2_eta = static_cast<float>(jet2_eta);
        out_dPhi_j2_mumu = static_cast<float>(dPhi_j2_mumu);

        out_pT_jj = static_cast<float>(pT_jj);
        out_Y_jj = static_cast<float>(Y_jj);
        out_dPhi_jj_mumu = static_cast<float>(dPhi_jj_mumu);

        out_m_jj = static_cast<float>(m_jj);
        out_met = met;
        out_HT = static_cast<float>(HT);

        out_m_mumu = static_cast<float>(m_mumu);
        out_eventNumber = eventNumber;
        out_jet_n = jet_n;
        out_channelNumber = channelNumber;
        out_eventWeight = eventWeight;

        outputTree->Fill();

        nSelected++;
        sumEventWeight += eventWeight;

        if (eventWeight > 0.0) nPositive++;
        else if (eventWeight < 0.0) nNegative++;
        else nZero++;
    }

    file->Close();
    delete file;
}


// ============================================================
// Main
// ============================================================

void build_nj2plus_dataset()
{
    // --------------------------------------------------------
    // Output variables
    // --------------------------------------------------------

    float pT_mumu;
    float Y_mumu;
    float cosTheta;

    float jet1_pt;
    float jet1_eta;
    float dPhi_j1_mumu;

    float jet2_pt;
    float jet2_eta;
    float dPhi_j2_mumu;

    float pT_jj;
    float Y_jj;
    float dPhi_jj_mumu;

    float m_jj;
    float met;
    float HT;

    float m_mumu;

    ULong64_t eventNumber;
    int jet_n;
    unsigned int channelNumber;
    double eventWeight;


    // --------------------------------------------------------
    // Signal output
    // --------------------------------------------------------

    TFile* signalFile =
        TFile::Open("nj2_signal.root", "RECREATE");

    TTree* signalTree =
        new TTree("signal", "nj >= 2 signal");


    signalTree->Branch("pT_mumu", &pT_mumu, "pT_mumu/F");
    signalTree->Branch("Y_mumu", &Y_mumu, "Y_mumu/F");
    signalTree->Branch("cosTheta", &cosTheta, "cosTheta/F");

    signalTree->Branch("jet1_pt", &jet1_pt, "jet1_pt/F");
    signalTree->Branch("jet1_eta", &jet1_eta, "jet1_eta/F");
    signalTree->Branch("dPhi_j1_mumu", &dPhi_j1_mumu, "dPhi_j1_mumu/F");

    signalTree->Branch("jet2_pt", &jet2_pt, "jet2_pt/F");
    signalTree->Branch("jet2_eta", &jet2_eta, "jet2_eta/F");
    signalTree->Branch("dPhi_j2_mumu", &dPhi_j2_mumu, "dPhi_j2_mumu/F");

    signalTree->Branch("pT_jj", &pT_jj, "pT_jj/F");
    signalTree->Branch("Y_jj", &Y_jj, "Y_jj/F");
    signalTree->Branch("dPhi_jj_mumu", &dPhi_jj_mumu, "dPhi_jj_mumu/F");

    signalTree->Branch("m_jj", &m_jj, "m_jj/F");
    signalTree->Branch("met", &met, "met/F");
    signalTree->Branch("HT", &HT, "HT/F");

    signalTree->Branch("m_mumu", &m_mumu, "m_mumu/F");
    signalTree->Branch("eventNumber", &eventNumber, "eventNumber/l");
    signalTree->Branch("jet_n", &jet_n, "jet_n/I");
    signalTree->Branch("channelNumber", &channelNumber, "channelNumber/i");
    signalTree->Branch("eventWeight", &eventWeight, "eventWeight/D");


    Long64_t sigSelected = 0;
    Long64_t sigPositive = 0;
    Long64_t sigNegative = 0;
    Long64_t sigZero = 0;
    Long64_t sigNonFinite = 0;
    Long64_t sigBadJetVectors = 0;
    double sigSumWeight = 0.0;


    ProcessFile(
        "ggH.root", signalTree,
        sigSelected, sigPositive, sigNegative, sigZero,
        sigNonFinite, sigBadJetVectors, sigSumWeight,
        pT_mumu, Y_mumu, cosTheta,
        jet1_pt, jet1_eta, dPhi_j1_mumu,
        jet2_pt, jet2_eta, dPhi_j2_mumu,
        pT_jj, Y_jj, dPhi_jj_mumu,
        m_jj, met, HT,
        m_mumu, eventNumber, jet_n, channelNumber, eventWeight
    );

    ProcessFile(
        "ggZH.root", signalTree,
        sigSelected, sigPositive, sigNegative, sigZero,
        sigNonFinite, sigBadJetVectors, sigSumWeight,
        pT_mumu, Y_mumu, cosTheta,
        jet1_pt, jet1_eta, dPhi_j1_mumu,
        jet2_pt, jet2_eta, dPhi_j2_mumu,
        pT_jj, Y_jj, dPhi_jj_mumu,
        m_jj, met, HT,
        m_mumu, eventNumber, jet_n, channelNumber, eventWeight
    );

    ProcessFile(
        "VBF.root", signalTree,
        sigSelected, sigPositive, sigNegative, sigZero,
        sigNonFinite, sigBadJetVectors, sigSumWeight,
        pT_mumu, Y_mumu, cosTheta,
        jet1_pt, jet1_eta, dPhi_j1_mumu,
        jet2_pt, jet2_eta, dPhi_j2_mumu,
        pT_jj, Y_jj, dPhi_jj_mumu,
        m_jj, met, HT,
        m_mumu, eventNumber, jet_n, channelNumber, eventWeight
    );


    signalFile->cd();
    signalTree->Write();
    signalFile->Close();


    // --------------------------------------------------------
    // Background output
    // --------------------------------------------------------

    TFile* backgroundFile =
        TFile::Open("nj2_background.root", "RECREATE");

    TTree* backgroundTree =
        new TTree("background", "nj >= 2 background");


    backgroundTree->Branch("pT_mumu", &pT_mumu, "pT_mumu/F");
    backgroundTree->Branch("Y_mumu", &Y_mumu, "Y_mumu/F");
    backgroundTree->Branch("cosTheta", &cosTheta, "cosTheta/F");

    backgroundTree->Branch("jet1_pt", &jet1_pt, "jet1_pt/F");
    backgroundTree->Branch("jet1_eta", &jet1_eta, "jet1_eta/F");
    backgroundTree->Branch("dPhi_j1_mumu", &dPhi_j1_mumu, "dPhi_j1_mumu/F");

    backgroundTree->Branch("jet2_pt", &jet2_pt, "jet2_pt/F");
    backgroundTree->Branch("jet2_eta", &jet2_eta, "jet2_eta/F");
    backgroundTree->Branch("dPhi_j2_mumu", &dPhi_j2_mumu, "dPhi_j2_mumu/F");

    backgroundTree->Branch("pT_jj", &pT_jj, "pT_jj/F");
    backgroundTree->Branch("Y_jj", &Y_jj, "Y_jj/F");
    backgroundTree->Branch("dPhi_jj_mumu", &dPhi_jj_mumu, "dPhi_jj_mumu/F");

    backgroundTree->Branch("m_jj", &m_jj, "m_jj/F");
    backgroundTree->Branch("met", &met, "met/F");
    backgroundTree->Branch("HT", &HT, "HT/F");

    backgroundTree->Branch("m_mumu", &m_mumu, "m_mumu/F");
    backgroundTree->Branch("eventNumber", &eventNumber, "eventNumber/l");
    backgroundTree->Branch("jet_n", &jet_n, "jet_n/I");
    backgroundTree->Branch("channelNumber", &channelNumber, "channelNumber/i");
    backgroundTree->Branch("eventWeight", &eventWeight, "eventWeight/D");


    Long64_t bkgSelected = 0;
    Long64_t bkgPositive = 0;
    Long64_t bkgNegative = 0;
    Long64_t bkgZero = 0;
    Long64_t bkgNonFinite = 0;
    Long64_t bkgBadJetVectors = 0;
    double bkgSumWeight = 0.0;


    ProcessFile(
        "Zskimmed_nj2.root", backgroundTree,
        bkgSelected, bkgPositive, bkgNegative, bkgZero,
        bkgNonFinite, bkgBadJetVectors, bkgSumWeight,
        pT_mumu, Y_mumu, cosTheta,
        jet1_pt, jet1_eta, dPhi_j1_mumu,
        jet2_pt, jet2_eta, dPhi_j2_mumu,
        pT_jj, Y_jj, dPhi_jj_mumu,
        m_jj, met, HT,
        m_mumu, eventNumber, jet_n, channelNumber, eventWeight
    );


    backgroundFile->cd();
    backgroundTree->Write();
    backgroundFile->Close();


    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------

    std::cout << "\n============================================\n";
    std::cout << "           nj >= 2 DATASET SUMMARY\n";
    std::cout << "============================================\n";

    std::cout << "\nSignal\n";
    std::cout << "Selected events       = " << sigSelected << "\n";
    std::cout << "Positive weights      = " << sigPositive << "\n";
    std::cout << "Negative weights      = " << sigNegative << "\n";
    std::cout << "Zero weights          = " << sigZero << "\n";
    std::cout << "Non-finite rejected   = " << sigNonFinite << "\n";
    std::cout << "Bad jet vectors       = " << sigBadJetVectors << "\n";
    std::cout << "Sum(eventWeight)      = " << sigSumWeight << "\n";

    std::cout << "\nBackground\n";
    std::cout << "Selected events       = " << bkgSelected << "\n";
    std::cout << "Positive weights      = " << bkgPositive << "\n";
    std::cout << "Negative weights      = " << bkgNegative << "\n";
    std::cout << "Zero weights          = " << bkgZero << "\n";
    std::cout << "Non-finite rejected   = " << bkgNonFinite << "\n";
    std::cout << "Bad jet vectors       = " << bkgBadJetVectors << "\n";
    std::cout << "Sum(eventWeight)      = " << bkgSumWeight << "\n";

    std::cout << "\nCreated:\n";
    std::cout << "  nj2_signal.root\n";
    std::cout << "  nj2_background.root\n";
    std::cout << "============================================\n";
}