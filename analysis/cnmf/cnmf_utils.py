def get_anatomy_short(anatomy, debug=False):
    ''' 
    Get the shortened anatomical name 
    '''
    anat_dict = {
        'ctx_lh_Unknown': 'Unknown',
        'ctx_lh_G_and_S_frontomargin': '',
        'ctx_lh_G_and_S_occipital_inf': '',
        'ctx_lh_G_and_S_paracentral': '',
        'ctx_lh_G_and_S_subcentral': 'L-subcentral',
        'ctx_lh_G_and_S_transv_frontopol': '',
        'ctx_lh_G_and_S_cingul-Ant': 'L-cing',
        'ctx_lh_G_and_S_cingul-Mid-Ant': 'L-cing',
        'ctx_lh_G_and_S_cingul-Mid-Post': 'L-pcing',
        'ctx_lh_G_cingul-Post-dorsal': 'L-cing',
        'ctx_lh_G_cingul-Post-ventral': 'L-cing',
        'ctx_lh_G_cuneus': 'L-cuneus',
        'ctx_lh_G_front_inf-Opercular': 'L-IFG-op',
        'ctx_lh_G_front_inf-Orbital': 'L-IFG-orb',
        'ctx_lh_G_front_inf-Triangul': 'L-IFG-tri',
        'ctx_lh_G_front_middle': 'L-MFG',
        'ctx_lh_G_front_sup': 'L-SFG',
        'ctx_lh_G_Ins_lg_and_S_cent_ins': 'L-ins',
        'ctx_lh_G_insular_short': 'L-ins-short',
        'ctx_lh_G_occipital_middle': '',
        'ctx_lh_G_occipital_sup': '',
        'ctx_lh_G_oc-temp_lat-fusifor': '',
        'ctx_lh_G_oc-temp_med-Lingual': '',
        'ctx_lh_G_oc-temp_med-Parahip': '',
        'ctx_lh_G_orbital': 'L-orbG',
        'ctx_lh_G_pariet_inf-Angular': 'L-angG',
        'ctx_lh_G_pariet_inf-Supramar': 'L-supramargG',
        'ctx_lh_G_parietal_sup': 'L-supPariet',
        'ctx_lh_G_postcentral': 'L-postCG',
        'ctx_lh_G_precentral': 'L-preCG',
        'ctx_lh_G_precuneus': 'L-precuneus',
        'ctx_lh_G_rectus': '',
        'ctx_lh_G_subcallosal': '',
        'ctx_lh_G_temp_sup-G_T_transv': 'L-HG',
        'ctx_lh_G_temp_sup-Lateral': 'L-STG',
        'ctx_lh_G_temp_sup-Plan_polar': 'L-PP',
        'ctx_lh_G_temp_sup-Plan_tempo': 'L-PT',
        'ctx_lh_G_temporal_inf': 'L-ITG',
        'ctx_lh_G_temporal_middle': 'L-MTG',
        'ctx_lh_Lat_Fis-ant-Horizont': 'L-pSTG',
        'ctx_lh_Lat_Fis-ant-Vertical': 'L-pSTG',
        'ctx_lh_Lat_Fis-post': 'L-pSTG',
        'ctx_lh_Medial_wall': '',
        'ctx_lh_Pole_occipital': '',
        'ctx_lh_Pole_temporal': '',
        'ctx_lh_S_calcarine': '',
        'ctx_lh_S_central': 'L-centralsulcus',
        'ctx_lh_S_cingul-Marginalis': '',
        'ctx_lh_S_circular_insula_ant': 'L-ins-circ-ant',
        'ctx_lh_S_circular_insula_inf': 'L-ins-circ-inf',
        'ctx_lh_S_circular_insula_sup': 'L-ins-circ-sup',
        'ctx_lh_S_collat_transv_ant': '',
        'ctx_lh_S_collat_transv_post': '',
        'ctx_lh_S_front_inf': 'L-IFS',
        'ctx_lh_S_front_middle': 'L-MFS',
        'ctx_lh_S_front_sup': 'L-SFS',
        'ctx_lh_S_interm_prim-Jensen': '',
        'ctx_lh_S_intrapariet_and_P_trans': '',
        'ctx_lh_S_oc_middle_and_Lunatus': '',
        'ctx_lh_S_oc_sup_and_transversal': '',
        'ctx_lh_S_occipital_ant': '',
        'ctx_lh_S_oc-temp_lat': '',
        'ctx_lh_S_oc-temp_med_and_Lingual': '',
        'ctx_lh_S_orbital_lateral': 'L-orb-sulc',
        'ctx_lh_S_orbital_med-olfact': 'L-orb-sulc',
        'ctx_lh_S_orbital-H_Shaped': 'L-orb-sulc',
        'ctx_lh_S_parieto_occipital': '',
        'ctx_lh_S_pericallosal': '',
        'ctx_lh_S_postcentral': 'L-postCS',
        'ctx_lh_S_precentral-inf-part': 'L-preCS-inf',
        'ctx_lh_S_precentral-sup-part': 'L-preCS-sup',
        'ctx_lh_S_suborbital': '',
        'ctx_lh_S_subparietal': '',
        'ctx_lh_S_temporal_inf': 'L-ITS',
        'ctx_lh_S_temporal_sup': 'L-STS',
        'ctx_lh_S_temporal_transverse': 'L-TTS',
        'ctx_rh_Unknown': '',
        'ctx_rh_G_and_S_frontomargin': '',
        'ctx_rh_G_and_S_occipital_inf': '',
        'ctx_rh_G_and_S_paracentral': '',
        'ctx_rh_G_and_S_subcentral': 'R-subcentral',
        'ctx_rh_G_and_S_transv_frontopol': '',
        'ctx_rh_G_and_S_cingul-Ant': 'R-cing',
        'ctx_rh_G_and_S_cingul-Mid-Ant': 'R-cing',
        'ctx_rh_G_and_S_cingul-Mid-Post': 'R-pcing',
        'ctx_rh_G_cingul-Post-dorsal': 'R-cing',
        'ctx_rh_G_cingul-Post-ventral': 'R-cing',
        'ctx_rh_G_cuneus': 'R-cuneus',
        'ctx_rh_G_front_inf-Opercular': 'R-IFG-op',
        'ctx_rh_G_front_inf-Orbital': 'R-IFG-orb',
        'ctx_rh_G_front_inf-Triangul': 'R-IFG-tri',
        'ctx_rh_G_front_middle': 'R-MFG',
        'ctx_rh_G_front_sup': 'R-SFG',
        'ctx_rh_G_Ins_lg_and_S_cent_ins': 'R-ins',
        'ctx_rh_G_insular_short': 'R-ins-short',
        'ctx_rh_G_occipital_middle': '',
        'ctx_rh_G_occipital_sup': '',
        'ctx_rh_G_oc-temp_lat-fusifor': '',
        'ctx_rh_G_oc-temp_med-Lingual': '',
        'ctx_rh_G_oc-temp_med-Parahip': '',
        'ctx_rh_G_orbital': 'R-orbG',
        'ctx_rh_G_pariet_inf-Angular': 'R-angG',
        'ctx_rh_G_pariet_inf-Supramar': 'R-supramargG',
        'ctx_rh_G_parietal_sup': 'R-supPariet',
        'ctx_rh_G_postcentral': 'R-postCG',
        'ctx_rh_G_precentral': 'R-preCG',
        'ctx_rh_G_precuneus': 'R-precuneus',
        'ctx_rh_G_rectus': '',
        'ctx_rh_G_subcallosal': '',
        'ctx_rh_G_temp_sup-G_T_transv': 'R-HG',
        'ctx_rh_G_temp_sup-Lateral': 'R-STG',
        'ctx_rh_G_temp_sup-Plan_polar': 'R-PP',
        'ctx_rh_G_temp_sup-Plan_tempo': 'R-PT',
        'ctx_rh_G_temporal_inf': 'R-ITG',
        'ctx_rh_G_temporal_middle': 'R-MTG',
        'ctx_rh_Lat_Fis-ant-Horizont': 'R-pSTG',
        'ctx_rh_Lat_Fis-ant-Vertical': 'R-pSTG',
        'ctx_rh_Lat_Fis-post': 'R-pSTG',
        'ctx_rh_Medial_wall': '',
        'ctx_rh_Pole_occipital': '',
        'ctx_rh_Pole_temporal': '',
        'ctx_rh_S_calcarine': '',
        'ctx_rh_S_central': 'R-centralsulcus',
        'ctx_rh_S_cingul-Marginalis': '',
        'ctx_rh_S_circular_insula_ant': 'R-ins-circ-ant',
        'ctx_rh_S_circular_insula_inf': 'R-ins-circ-inf',
        'ctx_rh_S_circular_insula_sup': 'R-ins-circ-sup',
        'ctx_rh_S_collat_transv_ant': '',
        'ctx_rh_S_collat_transv_post': '',
        'ctx_rh_S_front_inf': 'R-IFS',
        'ctx_rh_S_front_middle': 'R-MFS',
        'ctx_rh_S_front_sup': 'R-SFS',
        'ctx_rh_S_interm_prim-Jensen': '',
        'ctx_rh_S_intrapariet_and_P_trans': '',
        'ctx_rh_S_oc_middle_and_Lunatus': '',
        'ctx_rh_S_oc_sup_and_transversal': '',
        'ctx_rh_S_occipital_ant': '',
        'ctx_rh_S_oc-temp_lat': '',
        'ctx_rh_S_oc-temp_med_and_Lingual': '',
        'ctx_rh_S_orbital_lateral': '',
        'ctx_rh_S_orbital_med-olfact': '',
        'ctx_rh_S_orbital-H_Shaped': '',
        'ctx_rh_S_parieto_occipital': '',
        'ctx_rh_S_pericallosal': '',
        'ctx_rh_S_postcentral': 'R-postCS',
        'ctx_rh_S_precentral-inf-part': 'R-preCS-inf',
        'ctx_rh_S_precentral-sup-part': 'R-preCS-sup',
        'ctx_rh_S_suborbital': '',
        'ctx_rh_S_subparietal': '',
        'ctx_rh_S_temporal_inf': 'R-ITS',
        'ctx_rh_S_temporal_sup': 'R-STS',
        'ctx_rh_S_temporal_transverse': 'R-TTS',
        'wm_lh_Unknown': '',
        'wm_lh_G_and_S_frontomargin': '',
        'wm_lh_G_and_S_occipital_inf': '',
        'wm_lh_G_and_S_paracentral': '',
        'wm_lh_G_and_S_subcentral': '',
        'wm_lh_G_and_S_transv_frontopol': '',
        'wm_lh_G_and_S_cingul-Ant': '',
        'wm_lh_G_and_S_cingul-Mid-Ant': '',
        'wm_lh_G_and_S_cingul-Mid-Post': '',
        'wm_lh_G_cingul-Post-dorsal': '',
        'wm_lh_G_cingul-Post-ventral': '',
        'wm_lh_G_cuneus': '',
        'wm_lh_G_front_inf-Opercular': '',
        'wm_lh_G_front_inf-Orbital': '',
        'wm_lh_G_front_inf-Triangul': '',
        'wm_lh_G_front_middle': '',
        'wm_lh_G_front_sup': '',
        'wm_lh_G_Ins_lg_and_S_cent_ins': '',
        'wm_lh_G_insular_short': '',
        'wm_lh_G_occipital_middle': '',
        'wm_lh_G_occipital_sup': '',
        'wm_lh_G_oc-temp_lat-fusifor': '',
        'wm_lh_G_oc-temp_med-Lingual': '',
        'wm_lh_G_oc-temp_med-Parahip': '',
        'wm_lh_G_orbital': '',
        'wm_lh_G_pariet_inf-Angular': '',
        'wm_lh_G_pariet_inf-Supramar': '',
        'wm_lh_G_parietal_sup': '',
        'wm_lh_G_postcentral': '',
        'wm_lh_G_precentral': '',
        'wm_lh_G_precuneus': '',
        'wm_lh_G_rectus': '',
        'wm_lh_G_subcallosal': '',
        'wm_lh_G_temp_sup-G_T_transv': '',
        'wm_lh_G_temp_sup-Lateral': '',
        'wm_lh_G_temp_sup-Plan_polar': '',
        'wm_lh_G_temp_sup-Plan_tempo': '',
        'wm_lh_G_temporal_inf': '',
        'wm_lh_G_temporal_middle': '',
        'wm_lh_Lat_Fis-ant-Horizont': '',
        'wm_lh_Lat_Fis-ant-Vertical': '',
        'wm_lh_Lat_Fis-post': '',
        'wm_lh_Medial_wall': '',
        'wm_lh_Pole_occipital': '',
        'wm_lh_Pole_temporal': '',
        'wm_lh_S_calcarine': '',
        'wm_lh_S_central': '',
        'wm_lh_S_cingul-Marginalis': '',
        'wm_lh_S_circular_insula_ant': '',
        'wm_lh_S_circular_insula_inf': '',
        'wm_lh_S_circular_insula_sup': '',
        'wm_lh_S_collat_transv_ant': '',
        'wm_lh_S_collat_transv_post': '',
        'wm_lh_S_front_inf': '',
        'wm_lh_S_front_middle': '',
        'wm_lh_S_front_sup': '',
        'wm_lh_S_interm_prim-Jensen': '',
        'wm_lh_S_intrapariet_and_P_trans': '',
        'wm_lh_S_oc_middle_and_Lunatus': '',
        'wm_lh_S_oc_sup_and_transversal': '',
        'wm_lh_S_occipital_ant': '',
        'wm_lh_S_oc-temp_lat': '',
        'wm_lh_S_oc-temp_med_and_Lingual': '',
        'wm_lh_S_orbital_lateral': '',
        'wm_lh_S_orbital_med-olfact': '',
        'wm_lh_S_orbital-H_Shaped': '',
        'wm_lh_S_parieto_occipital': '',
        'wm_lh_S_pericallosal': '',
        'wm_lh_S_postcentral': '',
        'wm_lh_S_precentral-inf-part': '',
        'wm_lh_S_precentral-sup-part': '',
        'wm_lh_S_suborbital': '',
        'wm_lh_S_subparietal': '',
        'wm_lh_S_temporal_inf': '',
        'wm_lh_S_temporal_sup': '',
        'wm_lh_S_temporal_transverse': '',
        
        'wm_rh_Unknown': '',
        'wm_rh_G_and_S_frontomargin': '',
        'wm_rh_G_and_S_occipital_inf': '',
        'wm_rh_G_and_S_paracentral': '',
        'wm_rh_G_and_S_subcentral': '',
        'wm_rh_G_and_S_transv_frontopol': '',
        'wm_rh_G_and_S_cingul-Ant': '',
        'wm_rh_G_and_S_cingul-Mid-Ant': '',
        'wm_rh_G_and_S_cingul-Mid-Post': '',
        'wm_rh_G_cingul-Post-dorsal': '',
        'wm_rh_G_cingul-Post-ventral': '',
        'wm_rh_G_cuneus': '',
        'wm_rh_G_front_inf-Opercular': '',
        'wm_rh_G_front_inf-Orbital': '',
        'wm_rh_G_front_inf-Triangul': '',
        'wm_rh_G_front_middle': '',
        'wm_rh_G_front_sup': '',
        'wm_rh_G_Ins_lg_and_S_cent_ins': '',
        'wm_rh_G_insular_short': '',
        'wm_rh_G_occipital_middle': '',
        'wm_rh_G_occipital_sup': '',
        'wm_rh_G_oc-temp_lat-fusifor': '',
        'wm_rh_G_oc-temp_med-Lingual': '',
        'wm_rh_G_oc-temp_med-Parahip': '',
        'wm_rh_G_orbital': '',
        'wm_rh_G_pariet_inf-Angular': '',
        'wm_rh_G_pariet_inf-Supramar': '',
        'wm_rh_G_parietal_sup': '',
        'wm_rh_G_postcentral': '',
        'wm_rh_G_precentral': '',
        'wm_rh_G_precuneus': '',
        'wm_rh_G_rectus': '',
        'wm_rh_G_subcallosal': '',
        'wm_rh_G_temp_sup-G_T_transv': '',
        'wm_rh_G_temp_sup-Lateral': '',
        'wm_rh_G_temp_sup-Plan_polar': '',
        'wm_rh_G_temp_sup-Plan_tempo': '',
        'wm_rh_G_temporal_inf': '',
        'wm_rh_G_temporal_middle': '',
        'wm_rh_Lat_Fis-ant-Horizont': '',
        'wm_rh_Lat_Fis-ant-Vertical': '',
        'wm_rh_Lat_Fis-post': '',
        'wm_rh_Medial_wall': '',
        'wm_rh_Pole_occipital': '',
        'wm_rh_Pole_temporal': '',
        'wm_rh_S_calcarine': '',
        'wm_rh_S_central': '',
        'wm_rh_S_cingul-Marginalis': '',
        'wm_rh_S_circular_insula_ant': '',
        'wm_rh_S_circular_insula_inf': '',
        'wm_rh_S_circular_insula_sup': '',
        'wm_rh_S_collat_transv_ant': '',
        'wm_rh_S_collat_transv_post': '',
        'wm_rh_S_front_inf': '',
        'wm_rh_S_front_middle': '',
        'wm_rh_S_front_sup': '',
        'wm_rh_S_interm_prim-Jensen': '',
        'wm_rh_S_intrapariet_and_P_trans': '',
        'wm_rh_S_oc_middle_and_Lunatus': '',
        'wm_rh_S_oc_sup_and_transversal': '',
        'wm_rh_S_occipital_ant': '',
        'wm_rh_S_oc-temp_lat': '',
        'wm_rh_S_oc-temp_med_and_Lingual': '',
        'wm_rh_S_orbital_lateral': '',
        'wm_rh_S_orbital_med-olfact': '',
        'wm_rh_S_orbital-H_Shaped': '',
        'wm_rh_S_parieto_occipital': '',
        'wm_rh_S_pericallosal': '',
        'wm_rh_S_postcentral': '',
        'wm_rh_S_precentral-inf-part': '',
        'wm_rh_S_precentral-sup-part': '',
        'wm_rh_S_suborbital': '',
        'wm_rh_S_subparietal': '',
        'wm_rh_S_temporal_inf': '',
        'wm_rh_S_temporal_sup': '',
        'wm_rh_S_temporal_transverse': '',
        'Right-Cerebral-White-Matter': 'R-WM',
        'Left-Cerebral-White-Matter': 'L-WM',
        'Left-Hippocampus': 'L-hippo',
        'Right-Hippocampus': 'R-hippo',
        'Left-Amygdala': 'L-amyg',
        'Right-Amygdala': 'R-amyg',
        'ctx_rh_G_temporal_sup-Lateral': 'R-STG',
        'ctx_lh_G_temporal_sup-Lateral': 'L-STG',
    }
    if (anatomy in anat_dict.keys()) and (anat_dict[anatomy] != ''):
        anatomy_short = anat_dict[anatomy]
        if debug:
            print(f"{anatomy} --> {anatomy_short}")
    else:
        if debug:
            print(f"No short label for {anatomy}")
        anatomy_short = anatomy

    return anatomy_short


def tkRAS_to_MNI(elecmatrix, subj='cvs_avg35_inMNI152', imaging_dir='/Users/jsh3653/Box/ECoG_imaging'):
    '''
    Convert electrode coordinates from surface RAS (what's in TDT_elecs_all
    and TDT_elecs_all_warped.mat) to MNI RAS. This will allow you to 
    plot electrodes on the glass brain in nilearn.

    Inputs:
        elecmatrix [array] : [channels x 3] coordinates in tkRAS space. These are from
                             TDT_elecs_all.mat or TDT_elecs_all_warped.mat
        subj [str] : The subject space for these electrodes. For TDT_elecs_all_warped.mat,
                     this is *always* 'cvs_avg35_inMNI152', regardless of which subject
                     is being warped. If you need MNI RAS in the native space for some reason,
                     set subj = 'S0017' or similar
        imaging_dir [str] : path to your freesurfer subjects directory. Must have the whole
                            set of outputs from freesurfer to load MRIs, transforms, etc.

    Output:
        elecRAS [array] : [channels x 3] coordinates in MNI RAS space. These will be
                           voxel coordinates for plotting on an MRI, *not* on a surface.

    '''
    Vox2tkrRAS = [[-1.,  0.,  0.,  128.], 
              [ 0.,  0.,  1., -128.],
              [ 0., -1.,  0.,  128.],
              [ 0.,  0.,  0.,    1.]]

    if subj == 'cvs_avg35_inMNI152':
        xfm_t = [[ 1.054057,  0.002858, -0.007161, -0.076111],
                 [-0.010547,  1.091444,  0.009027,  1.779144],
                 [ 0.020196, -0.059968,  1.101421, -5.461380]]
        
        vox2ras = [[-1.00000,  0.00000, 0.00000,  127.00000], 
                   [ 0.00000,  0.00000, 1.00000, -145.00000], 
                   [ 0.00000, -1.00000, 0.00000,  147.00000], 
                   [ 0.00000,  0.00000, 0.00000,    1.00000]]
    else:
        mri_fname = f'{imaging_dir}/{subj}/mri/orig.mgz'
        mri = nib.load(mri_fname)
        vox2ras = mri.header.get_vox2ras()
        xfm = f'{imaging_dir}/{subj}/mri/transforms/talairach.xfm'
        xfm_t = np.array(np.loadtxt(xfm,skiprows=5,delimiter=' ',comments=';'))
    
    elecRAS = np.dot(xfm_t, np.dot(vox2ras, np.dot(np.linalg.inv(Vox2tkrRAS), 
                  np.hstack((elecmatrix, np.ones((elecmatrix.shape[0],1)))).T))).T[:,:3]

    return elecRAS

def sem(epochs):
    '''
    calculates standard error margin across epochs
    epochs should have shape (epochs,samples)
    '''
    sem_below = epochs.mean(0) - (epochs.std(0)/np.sqrt(epochs.shape[0]))
    sem_above = epochs.mean(0) + (epochs.std(0)/np.sqrt(epochs.shape[0]))
    return sem_below, sem_above

def pve(X, nmf):
    ''' Calculate the percent variance explained for a particular NMF solution.
    Note that if you include the same number of channels as clusters then you'll
    get 100% variance explained, so here we will set a cut off of 90%.
    
    Input:
        X [array]: Matrix of data [channels x time]
        nmf [object] : NMF solution from pymf3.convexNMF. 
    Output:
        pve [float] : Variance explained (between 0-1)
    '''
    Xnew = np.dot(nmf.W, nmf.H)
    
    # This is using the residual sum of squares over the total sum of
    # squares.  Percent variance explained is 1 minus this.
    # See https://en.wikipedia.org/wiki/Coefficient_of_determination
    # The mean across all X is 0 because of z-scoring, so denominator is
    # simplified
    pve = 1-np.sum((Xnew-X)**2)/np.sum(X**2)
    return pve