import plotting_utils as plotting
reload(plotting)
import pandas as pd
import root_pandas as rpd
import sys

filename = sys.argv[1]
#file_name='ttbar_RegressionPerJet_heppy_energyRings3_forTesting.root'

#inout_dir = '/users/nchernya/HHbbgg_ETH/root_files/'
inout_dir = '/mnt/t3nfs01/data01/shome/nchernya/HHbbgg_ETH_devel/root_files/heppy_05_10_2017/'
filename = inout_dir+filename
print filename

branch_names = 'Jet_pt,Jet_eta,Jet_mcFlavour,Jet_mcPt,rho,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel,Jet_leptonDeltaR,Jet_neHEF,Jet_neEmEF,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,Jet_vtx3deL,Jet_energyRing_dR0_em_Jet_e,Jet_energyRing_dR1_em_Jet_e,Jet_energyRing_dR2_em_Jet_e,Jet_energyRing_dR3_em_Jet_e,Jet_energyRing_dR4_em_Jet_e,Jet_energyRing_dR0_neut_Jet_e,Jet_energyRing_dR1_neut_Jet_e,Jet_energyRing_dR2_neut_Jet_e,Jet_energyRing_dR3_neut_Jet_e,Jet_energyRing_dR4_neut_Jet_e,Jet_energyRing_dR0_ch_Jet_e,Jet_energyRing_dR1_ch_Jet_e,Jet_energyRing_dR2_ch_Jet_e,Jet_energyRing_dR3_ch_Jet_e,Jet_energyRing_dR4_ch_Jet_e,Jet_energyRing_dR0_mu_Jet_e,Jet_energyRing_dR1_mu_Jet_e,Jet_energyRing_dR2_mu_Jet_e,Jet_energyRing_dR3_mu_Jet_e,Jet_energyRing_dR4_mu_Jet_e,Jet_numDaughters_pt03,Jet_pt_reg'.split(",")
cuts='(Jet_pt > 15) & (Jet_eta<2.5 & Jet_eta>-2.5) & (Jet_mcFlavour==5 | Jet_mcFlavour==-5) & (Jet_mcPt>0) & (Jet_mcPt<6000)' #all for presentaion on 23 Jan was done with pT cut at 20 GeV
tmp_data_frame_cut = (rpd.read_root(filename+'.root','tree', columns = branch_names)).query(cuts)
tmp_data_frame_cut.to_hdf(filename+'.hd5'  ,'w')
