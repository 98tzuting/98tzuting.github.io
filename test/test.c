#include "TFile.h"
#include "TTree.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <TH1D.h>
#include <TAttLine.h>
#include <TStyle.h>
using namespace std;

void test()
{
    auto c_Mbc = new TCanvas("c_Mbc", "BPRE");
    TFile *file = new TFile("~/SKIM/SKIM_eeppg_0.root");
    TTree *tree = (TTree *)file->Get("variables"); 
    int entries = (int)tree->GetEntries();
    TH1F *Bs0_Mbc = new TH1F("Bs0_Mbc","",150,5.25,5.38);
    Double_t Mbc;
    tree->SetBranchAddress("Mbc",&Mbc);
    for(int i =0;i<entries;i++)
    {
        tree->GetEntry(i);
        Bs0_Mbc->Fill(Mbc);
    }
    gStyle->SetOptStat(111111);
    gStyle->SetStatX(0.7);
    gStyle->SetStatW(0.2);
    gStyle->SetLabelOffset(1.2);
    Bs0_Mbc->SetFillColor(kBlue); 
    Bs0_Mbc->Draw(); 
    Bs0_Mbc->SetTitle("Mbc");
    Bs0_Mbc->GetXaxis()->SetTitle("Mbc(GeV)");
    Bs0_Mbc->GetYaxis()->SetTitle("Event");
    Bs0_Mbc->GetYaxis()->SetTitleFont(62); 
    Bs0_Mbc->GetXaxis()->SetTitleFont(62);

    c_Mbc->SaveAs("Mbc.png");
}

