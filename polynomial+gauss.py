import ROOT
import sys
import ROOT.RooFit as rf
from ROOT import (
    RooRealVar,
    RooChebychev,
    TChain,
    RooArgSet,
    RooDataSet,
    RooArgList,
    TCanvas,
    RooCBShape,
    RooGaussian,
    RooAddPdf,
    RooVoigtian,
    RooPolynomial,
)

filename = sys.argv[1]

def fitting():
    deltaE = RooRealVar("deltaE", "#Delta E", -0.2, 0.2)
    RootFile = TChain("variables")
    RootFile.Add(filename)
    VarSet = RooArgSet(deltaE)
    data = RooDataSet("data", "data", RootFile, VarSet)
    deltaE.setRange("FullRange", -0.2, 0.2)
    mean = RooRealVar('mean1', 'Mbc mean',0.0014, -0.2, 0.2)
    a1 = RooRealVar("a1","a1",1,-10,10)
    a2 = RooRealVar("a2","a2",0,-10,10)
    sigma = RooRealVar("Sigma", "Signal  #sigma 2", 0.0179, 0.0, 3.0)
    SG = RooGaussian("SG_Gau", "Signal Gaussian Shape", deltaE, mean, sigma)
    Poly = RooPolynomial("bkg", "bkg PDF",deltaE, RooArgList(a1,a2),1)
    fraction_1 = RooRealVar("SG_frac1", "Signal CrystalBall Fraction", 0.5, 0.0, 1.0)
    pdf_1 = RooAddPdf("SG_Pdf", "Signal PDF", RooArgList(SG, Poly), RooArgList(fraction_1))
        
    FitResult = pdf_1.fitTo(data, rf.Save(ROOT.kTRUE), rf.Range("SignalRange"))
  
    canvas = TCanvas("canvas", "c1", 1600, 900)
    canvas.Clear()
    plot = deltaE.frame(
        rf.Name("deltaE_plot"),
        rf.Title("deltaE_plot"),
        rf.Bins(40),
        rf.Range("FullRange"),
        rf.Save(ROOT.kTRUE),
    )
    data.plotOn(plot, rf.Range("FullRange"), rf.LineColor(ROOT.kRed))
    pdf_1.plotOn(plot)
    pdf_1.Print("t")
    pdf_1.paramOn(plot, rf.Layout(0.6))

    plot.Draw()
    canvas.SaveAs("eeppg2.png")
fitting()