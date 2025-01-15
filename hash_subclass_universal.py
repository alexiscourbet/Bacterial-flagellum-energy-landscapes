#Shall cover all pair-cases
#!/software/miniconda3/envs/pyrosetta3/bin/python

#Description: Hash Residue-pairs helper classes
#Ver: 0.1a
#Author: David Baker
#Author: Modified by Daniel-Adriano Silva
#Date: Nov,2017
### Currently used for pairs-of-hbonds, 
### but can be easily extended to any residue-pair-interaction

import os
import pyrosetta
import numpy as np

import collections

import rif.hash
from pyrosetta.rosetta.numeric import xyzVector_double_t as V3

    
# hash is from sc of sc_res to sc and bb of scbb_res.   key depends only on transform and possibly phi and psi of scbb_res.    Hence only require frame for res1, but pose for res2
# info stored may be res_names, chis of both residues, and relevant phipsi of scbb_res 
# WARN(onalant): THIS CLASS SHOULD NEVER BE EXPLICITLY INSTANTIATED
class aa_pair_hash_universal:
    
    def __init__(self,
                 tor_resl,
                 cart_resl, 
                 ori_resl, 
                 cart_bound):
        print("Generating pair-bb hashing function with parameters:", tor_resl,
                                                                      cart_resl, 
                                                                      ori_resl, 
                                                                      cart_bound)
        self.uses_phi=False
        self.uses_psi=False
        if ("phi" in self.dof_resB):
            self.uses_phi=True
        if ("psi" in self.dof_resB):
            self.uses_psi=True
        ##self.uses_phipsi=(self.uses_phi * self.uses_psi)
        
        #ToDo probably IF these functions based on uses_phi&psi
        self.stubHash_function = rif.hash.RosettaStubHash(cart_resl, ori_resl, cart_bound)
        self.oneTorsionStubHash_function = rif.hash.RosettaStubTorsionHash(tor_resl,cart_resl, ori_resl, cart_bound)
        self.twoTorsionStubHash_function = rif.hash.RosettaStubTwoTorsionHash(tor_resl, cart_resl, ori_resl, cart_bound)
        
        print("StubHash Grid:", self.stubHash_function.grid)
        print("StubTorsionHash Grid:", self.oneTorsionStubHash_function.grid)
        print("StubTwoTorsionHash Grid:", self.twoTorsionStubHash_function.grid)
        
    def get_bin_from_frame_residue(self, frame, scbb_res):
        bb_stub = pyrosetta.rosetta.core.kinematics.Stub(frame.global2local(scbb_res.xyz('N')), 
                                                         frame.global2local(scbb_res.xyz('CA')), 
                                                         frame.global2local(scbb_res.xyz('C'))
                                                        )
        return self.stubHash_function.get_key(bb_stub)
    
    def get_bin_from_frame_residue_oneAngle(self, frame, scbb_res, res_tor):
        bb_stub = pyrosetta.rosetta.core.kinematics.Stub(frame.global2local(scbb_res.xyz('N')), 
                                                         frame.global2local(scbb_res.xyz('CA')), 
                                                         frame.global2local(scbb_res.xyz('C'))
                                                         )
        return self.oneTorsionStubHash_function.get_key(bb_stub, res_tor)
   
    def get_bin_from_frame_residue_twoAngles(self, frame, scbb_res, res_torA, res_torB):
        bb_stub = pyrosetta.rosetta.core.kinematics.Stub(frame.global2local(scbb_res.xyz('N')), 
                                                         frame.global2local(scbb_res.xyz('CA')), 
                                                         frame.global2local(scbb_res.xyz('C'))
                                                         )
        return self.twoTorsionStubHash_function.get_key(bb_stub, res_torA, res_torB)
            
    def get_frame_from_res(self, res):
        return pyrosetta.rosetta.core.kinematics.Stub(V3(res.xyz('N')),V3(res.xyz('CA')),V3(res.xyz('C')))

# we are hashing sc to scbb interactions (mainly hbonds).  will keep sc_res fixed, and sample rigid body orientation and relevant backbone torsions of scbb_res.
# so just have to orient rotamers once onto sc res, and then keep them fixed. 
class make_hash_universal(aa_pair_hash_universal):
    
    def get_hash_entry_dtype(self):
        return np.dtype([("keyStub", np.uint64, 1),
                         ("keyTor", np.uint64, (1 * (self.uses_phi or self.uses_psi))),
                         ("ener",  np.int8, 1),
                         ("phiA",  np.int8, self.num_phiA),
                         ("psiA",  np.int8, self.num_psiA),
                         ("phiB",  np.int8, self.num_phiB),
                         ("psiB",  np.int8, self.num_psiB),
                         ("chisA", np.int8, self.num_chisA),
                         ("chisB", np.int8, self.num_chisB)])
    
    def bringAngleToOneEightyDomain(self,
                                    a):
        return ((a-180.)%360.)-180.
    
    def __init__(self,
                 in_pose,
                 resAndx,
                 resBndx,
                 bbConfAnglesA=[],
                 bbConfAnglesB=[],
                 orient_atomsA=[],
                 orient_atomsB=[],
                 dof_resA=[],
                 dof_resB=[],
                 tor_resl=10.0,
                 cart_resl=1.0,
                 ori_resl=10,
                 cart_bound=32,
                 threshold_energy_for_hashing=0.0,
                 energy_sfx=False):
        
        self.dd=collections.defaultdict(list)
        
        self.in_pose_copy=in_pose.clone()
        self.resA=in_pose.residue(resAndx)
        self.resB=in_pose.residue(resBndx)
        self.resAndx=resAndx
        self.resBndx=resBndx
        self.bbConfAnglesA=bbConfAnglesA
        self.bbConfAnglesB=bbConfAnglesB
        self.orient_atomsA=orient_atomsA
        self.orient_atomsB=orient_atomsB
        self.dof_resA=dof_resA
        self.dof_resB=dof_resB
        
        #Pose and SFX To Score Pairs of Residues
        self.threshold_energy_for_hashing=threshold_energy_for_hashing
        print("Hashing with energy limit of: ", self.threshold_energy_for_hashing)
        print("Creating score_pair_pose")
        self.scoring_pair_pose=pyrosetta.pose_from_sequence("A")
        tmp_scoring_pair_poseB=pyrosetta.pose_from_sequence("A")
        self.scoring_pair_pose.append_pose_by_jump(tmp_scoring_pair_poseB,1)
        if not(energy_sfx):
            self.sfxPair=pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function("beta")
        else:
            self.sfxPair=energy_sfx
        
        super(make_hash_universal, self).__init__(tor_resl,
                                                  cart_resl, 
                                                  ori_resl, 
                                                  cart_bound)
        #General counter
        self.n_added=0
        
        #Populate the size of the entry in the hash dictionary for resA
        #simplified_dofs_A=np.asarray([x[:3] for x in dof_resA])
        self.num_phiA=1*("phi" in self.dof_resA) if len(self.dof_resA) else 0
        self.num_psiA=1*("psi" in self.dof_resA) if len(self.dof_resA) else 0
        self.sample_inverseRot_A=(self.num_phiA+self.num_psiA)<=0 #Determine if sampling inverse rotamers
        self.num_chisA=self.resA.nchi()
        
          
        #Populate the size of the entry in the hash dictionary for resB
        #simplified_dofs_B=np.asarray([x[:3] for x in dof_resB])
        self.num_phiB=1*("phi" in self.dof_resB) if len(self.dof_resB) else 0
        self.num_psiB=1*("psi" in self.dof_resB) if len(self.dof_resB) else 0
        self.sample_inverseRot_B=(self.num_phiB+self.num_psiB)<=0 #Determine if sampling inverse rotamers
        self.num_chisB=self.resB.nchi()
    
        #Add rotamers for A?
        if self.sample_inverseRot_A:
            print('Will sample inverse rotamers for resA')
            self.orient_vA=pyrosetta.rosetta.utility.vector1_std_pair_std_string_std_string_t()
            for atom in self.orient_atomsA:
                self.orient_vA.append( (atom, atom) )
            ###print("orientV A is:", orient_vA)

            #generate and store all inverse rotamers of ResA, their frames, and their chis            
            self.rotsA=self.get_rots(self.resA,
                                     bbConfAnglesA) 
            print('# of inverse rotamers used in hashing for resA',len(self.rotsA))

            # Generate info for rotamers of A
            self.rot_listA=[]
            # now add on info for library rotamers (First result is  for starting rotamer A)            
            for rot in self.rotsA:
                rot.orient_onto_residue(self.resA,
                                        self.orient_vA)
                frame=self.get_frame_from_res(rot)
                self.rot_listA.append({"rot" : rot, "frame" : frame}) #,"chis" : tuple(chi_listA)})

        #Add rotamers for B?
        if self.sample_inverseRot_B:
            print('Will sample inverse rotamers for resB')
            self.orient_vB=pyrosetta.rosetta.utility.vector1_std_pair_std_string_std_string_t()
            for atom in self.orient_atomsB:
                self.orient_vB.append( (atom, atom) )
                
            #generate and store all inverse rotamers of resB, their frames, and their chis            
            self.rotsB=self.get_rots(self.resB,
                                     bbConfAnglesA) 
            print('# of inverse rotamers used in hashing for resB',len(self.rotsB))
        
    def get_rots(self,
                 sc_res,
                 inverse_rot_backbones):
        sfx = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function("beta")
        rots=[sc_res.clone()] #Add initial/current to the beggining of the list
        for angles in inverse_rot_backbones:
            rots = rots + self.generate_canonical_rotamer_residues(sc_res.name3(),
                                                                   angles,
                                                                   sfx)
        return rots
        
    def generate_canonical_rotamer_residues(self,
                                            residue_name , 
                                            target_phi_psi,
                                            sfx):
        canonical_phi_psi = {"helical" : (-66.0,-42.0),
                             "sheet" : (-126.0,124.0)}
        
        #This is a nightmare of polymorphism
        if target_phi_psi in canonical_phi_psi:
            target_phi , target_psi = canonical_phi_psi[target_phi_psi]
            ##print(target_phi)
            ##print(target_psi)
        else:
            assert()
        
        #This TryRotamers is pure-madness, uses up to ex4, and you can't change it
        ##tryrot = TryRotamers(3, sfx, 0, 0, True, solo_res=False, include_current=False )
        test_pose = pyrosetta.rosetta.core.pose.Pose()
        
        test_sequence = ("A"*4)+("X[%s]")%residue_name+("A"*4) 
        #print(test_sequence)
        pyrosetta.rosetta.core.pose.make_pose_from_sequence( test_pose, test_sequence, "fa_standard" )
        
        #Make a curly pose :)
        for i in range(1,test_pose.size()+1):
            test_pose.set_psi(i, target_psi)
            test_pose.set_phi(i, target_phi)
            test_pose.set_omega(i, 180)
        
        #Debug
        ##test_pose.dump_pdb("/home/dadriano/tmp/testPoseRot_%s.pdb"%target_phi_psi)
        
        rotamer_list = self.get_rotamers_for_res_in_pose(test_pose,
                                                5,
                                                self.sfxPair,
                                                ex1=True,
                                                ex2=False,
                                                ex3=False,
                                                ex4=False,
                                                check_clashes=True)
        return rotamer_list

    def get_rotamers_for_res_in_pose(self,
                                 in_pose,
                                 target_residue,
                                 sfx,
                                 ex1=True,
                                 ex2=False,
                                 ex3=False,
                                 ex4=False,
                                 check_clashes=True):

        packTask=pyrosetta.rosetta.core.pack.task.TaskFactory.create_packer_task(in_pose)
        packTask.set_bump_check(check_clashes)

        resTask=packTask.nonconst_residue_task(target_residue)
        resTask.or_ex1( ex1 )
        resTask.or_ex2( ex2 )
        resTask.or_ex3( ex3 )
        resTask.or_ex4( ex4 )
        resTask.restrict_to_repacking();

        packer_graph=pyrosetta.rosetta.utility.graph.Graph( in_pose.size() ) ;


        rsf=pyrosetta.rosetta.core.pack.rotamer_set.RotamerSetFactory()
        rotset_ = rsf.create_rotamer_set( in_pose.residue( target_residue ) );
        rotset_.set_resid(target_residue)



        rotset_.build_rotamers(in_pose, 
                               sfx, 
                               packTask, 
                               packer_graph, 
                               False )

        print ("Found num_rotamers:", rotset_.num_rotamers())
        rotamer_set=[]
        for irot in range(1,rotset_.num_rotamers()+1):
            rotamer_set.append(rotset_.rotamer(irot).clone())
        return rotamer_set

    def update_hash(self,
                    pose):
        #NOTE: divided by 2 to fit into sInt4 -180 to 179
        #ToDo
        #1. Add case elif self.sample_inverseRot_B:
        #2. Add case else (i.e. scbb-scbb sampling, and also covers bb-bb)
        if self.sample_inverseRot_A:
            for rot_infoA in self.rot_listA: 
                #Calc inverse rotamers for B
                if self.sample_inverseRot_B: 
                    for rotB in self.rotsB:
                        #ToDo Add scoring to remove bad rotamer-pairs
                        rotB.orient_onto_residue(pose.residue(self.resBndx),
                                                self.orient_vB)
                        tmp_score=self.score_residue_pair(rot_infoA["rot"],
                                                             rotB)
                        if (tmp_score>self.threshold_energy_for_hashing):
                            continue
                            
                        #Create entry and add torsions
                        k=self.get_bin_from_frame_residue(rot_infoA['frame'],
                                                                   rotB)
                        k_tor=False
                        
                        """#This could be uncommented if there is a reason to store phi-psi of certain B-rot. Very unlikely and would need extra logic for rotB generation
                        if (self.uses_phi and self.uses_psi):
                            k_tor=self.get_bin_from_frame_residue_twoAngles(rot_infoA['frame'],
                                                                            rotB,
                                                                            res_torA=pose.phi(self.resBndx),
                                                                            res_torB=pose.psi(self.resBndx))
                        elif(self.uses_phi):
                            k_tor=self.get_bin_from_frame_residue_oneAngle(rot_infoA['frame'],
                                                                           rotB,
                                                                           res_tor=pose.phi(self.resBndx))
                        elif(self.uses_psi):
                            k_tor=self.get_bin_from_frame_residue_oneAngle(rot_infoA['frame'],
                                                                           rotB,
                                                                           res_tor=pose.psi(self.resBndx))
                        """
                        
                        self.dd[k,k_tor].append(np.zeros(1,self.get_hash_entry_dtype()))
                        self.dd[k,k_tor][-1]['keyStub']=k
                        
                        self.dd[k,k_tor][-1]['ener']=max(-127,min(tmp_score,127)) #Store the energy bounded to -127 +127
                        for jchi in range(self.num_chisA):
                            self.dd[k,k_tor][-1]['chisA'][0,jchi]=self.bringAngleToOneEightyDomain(rot_infoA["rot"].chi(jchi+1))/2
                        for jchi in range(self.num_chisB):
                            self.dd[k,k_tor][-1]['chisB'][0,jchi]=self.bringAngleToOneEightyDomain(rotB.chi(jchi+1))/2
                        #General counter
                        self.n_added+=1

                else:
                    tmp_score=self.score_residue_pair(rot_infoA["rot"],
                                                      pose.residue(self.resBndx))
                    if (tmp_score>self.threshold_energy_for_hashing):
                        ##self.scoring_pair_pose.dump_pdb("/home/dadriano/tmp/tmp/testClashRP%0.2f.pdb"%tmp_score)
                        continue
                    k=self.get_bin_from_frame_residue(rot_infoA["frame"],
                                                 pose.residue(self.resBndx))
                    k_tor=False
                    if (self.uses_phi and self.uses_psi):
                        k_tor=self.get_bin_from_frame_residue_twoAngles(rot_infoA["frame"],
                                                                        pose.residue(self.resBndx),
                                                                        res_torA=pose.phi(self.resBndx),
                                                                        res_torB=pose.psi(self.resBndx))
                    elif(self.uses_phi):
                        k_tor=self.get_bin_from_frame_residue_oneAngle(rot_infoA["frame"],
                                                                       pose.residue(self.resBndx),
                                                                       res_tor=pose.phi(self.resBndx))
                    elif(self.uses_psi):
                        k_tor=self.get_bin_from_frame_residue_oneAngle(rot_infoA["frame"],
                                                                       pose.residue(self.resBndx),
                                                                       res_tor=pose.psi(self.resBndx))
                    #Create entry and add torsions
                    self.dd[k,k_tor].append(np.zeros(1,self.get_hash_entry_dtype()))
                    self.dd[k,k_tor][-1]['keyStub']=k
                    if k_tor:
                        self.dd[k,k_tor][-1]['keyTor']=k_tor
                    self.dd[k,k_tor][-1]['ener']=max(-127,min(tmp_score,127)) #Store the energy bounded to -127 +127
                    for jchi in range(self.num_chisA):
                        self.dd[k,k_tor][-1]['chisA'][0,jchi]=self.bringAngleToOneEightyDomain(rot_infoA["rot"].chi(jchi+1))/2
                    if self.num_phiB: #instead you can use: self.uses_phi
                        self.dd[k,k_tor][-1]['phiB']=self.bringAngleToOneEightyDomain(pose.phi(self.resBndx))/2
                    if self.num_psiB: #instead you can use: self.uses_psi
                        self.dd[k,k_tor][-1]['psiB']=self.bringAngleToOneEightyDomain(pose.psi(self.resBndx))/2
                    for jchi in range(self.num_chisB):
                        self.dd[k,k_tor][-1]['chisB'][0,jchi]=self.bringAngleToOneEightyDomain(pose.residue(self.resBndx).chi(jchi+1))/2
                    #General counter
                    self.n_added+=1
                    
    def score_residue_pair(self,
                           resA,
                           resB):
        self.scoring_pair_pose.replace_residue( seqpos=1, 
                                          new_rsd_in=resA, 
                                          orient_backbone=False );
        self.scoring_pair_pose.replace_residue( seqpos=2, 
                                          new_rsd_in=resB, 
                                          orient_backbone=False );
        return(self.sfxPair(self.scoring_pair_pose))

    def dump_hash_into_nptable(self,
                               outname,
                               metadata):
        final_hash_table=np.zeros(self.n_added, self.get_hash_entry_dtype())
        tmp_counter=0
        for ikey in self.dd:
            for jentry in self.dd[ikey]:
                final_hash_table[tmp_counter]=jentry
                tmp_counter+=1
        assert(tmp_counter==self.n_added)
        print("Saving np table of sampling with size %d to %s"%(len(final_hash_table),outname))
        np.save(outname, [(self.resA.name3(),self.resB.name3()),
                           metadata, 
                           np.unique(final_hash_table)]) 
        print("Done Saving")
            


class use_hash_universal(aa_pair_hash_universal):
    class do_nothing_to_pose:
        def set_target(self,
                       input_obj):
            return
        def apply(self,
                  input_obj):
            return
    
    def __init__(self, 
                 inDirPath,
                 dbExtension=".hbdbpair.npy",
                 times_std_dev_threshold=10.0,#autocalculated min energy limit average +this-std
                 max_pair_energy=50, #min energy limit average +this-std
                 min_num_hbonds_pair=2, #min number of h-bonds in any given pair
                 nhbond_threshold_nbonus=1, #min nhbonds auto-calculated will be substracted from this number (because h-bonds with coarsed sidechains might not be perfect always.)
                 do_use_gly=False,
                 use_stub_for_all_tors=False
                ): #,hash_function,scbb_torsions,filename,res_names_in_hash,name_data=name_data_type(None,None)):
        print ("Reading hash database from path:", inDirPath)
        print("Looking for files with extension:", dbExtension)
        self.hashes_store={ "stub": collections.defaultdict(list), 
                            "phi": collections.defaultdict(list), 
                            "psi": collections.defaultdict(list), 
                            "phipsi": collections.defaultdict(list)}
        
        if use_stub_for_all_tors:
            print("Using STUB-hashes for all torsion methods")
            self.hash_functions={
                    "stub":   lambda frame, pose, resndx: self.get_bin_from_frame_residue(frame, pose.residue(resndx)),
                    "phi":    lambda frame, pose, resndx: self.get_bin_from_frame_residue(frame, pose.residue(resndx)),
                    "psi":    lambda frame, pose, resndx: self.get_bin_from_frame_residue(frame, pose.residue(resndx)),
                    "phipsi": lambda frame, pose, resndx: self.get_bin_from_frame_residue(frame, pose.residue(resndx)),
            }
        
        else:
            print("Using 4-method hashing for torsions ( i.e. stub | phi | psi | phipsi )")
            self.hash_functions={
                    "stub":   lambda frame, pose, resndx: self.get_bin_from_frame_residue(frame, pose.residue(resndx)),
                    "phi":    lambda frame, pose, resndx: self.get_bin_from_frame_residue_oneAngle(frame, pose.residue(resndx), pose.phi(resndx)),
                    "psi":    lambda frame, pose, resndx: self.get_bin_from_frame_residue_oneAngle(frame, pose.residue(resndx), pose.psi(resndx)),
                    "phipsi": lambda frame, pose, resndx: self.get_bin_from_frame_residue_twoAngles(frame, pose.residue(resndx), pose.phi(resndx), pose.psi(resndx))
            }

        
        
        self.hashes_aapairs=collections.defaultdict(list)
        self.historic_data_dic={}
        self.xy_hash_vs_size_dic={}
        self.hash_array_dic_chisA={}
        self.hash_array_dic_chisB={}
        self.num_chis={}
        self.aapairs_poseMutators={}
        self.hbondenergy_threshold={}
        self.nhbonds_threshold={}
        for filename in os.listdir(inDirPath):
            if (dbExtension in filename):
                dictionary_path ='%s/%s'%(inDirPath,filename)
                print("Reading database file #%d: %s"% (len(self.hashes_aapairs)+1, dictionary_path))
                [aaPair,
                 historic_data,
                 xy_hash_vs_size,
                 xy_hash_vs_size_after_merge,
                 hash_array]=np.load(dictionary_path)
                
                #If the same A.A. pair is twice in the database we should subindex it
                self.hashes_aapairs[aaPair].append(("%s_n%d"%(aaPair[0],len(self.hashes_aapairs[aaPair])),
                                                    "%s_n%d"%(aaPair[1],len(self.hashes_aapairs[aaPair]))))
                #Get the AApair key
                aaPair=self.hashes_aapairs[aaPair][-1]    
                #Init arrays
                self.aapairs_poseMutators[aaPair]=None
                self.hbondenergy_threshold[aaPair]=None
                self.nhbonds_threshold[aaPair]=None
                #Inti data-arrays
                self.xy_hash_vs_size_dic[aaPair]=xy_hash_vs_size
                self.historic_data_dic[aaPair]=historic_data
                self.hash_array_dic_chisA[aaPair]=hash_array["chisA"]
                self.hash_array_dic_chisB[aaPair]=hash_array["chisB"]
                #Set degrees of freedom
                has_phiB = (hash_array["phiB"].size > 0)
                has_psiB = (hash_array["psiB"].size > 0)
                
                if use_stub_for_all_tors:
                    for indx in range(hash_array.shape[0]):
                        if not(has_phiB or has_psiB):
                            self.hashes_store["stub"][hash_array["keyStub"][indx]].append([aaPair, indx])
                        elif has_phiB and has_psiB:
                            self.hashes_store["phipsi"][hash_array["keyStub"][indx]].append([aaPair,indx])
                        elif has_phiB:
                            self.hashes_store["phi"][hash_array["keyStub"][indx]].append([aaPair,indx])
                        elif has_psiB:
                            self.hashes_store["psi"][hash_array["keyStub"][indx]].append([aaPair, indx])
                        else:
                            print("There is a mistake, this should never happen!")
                            assert()
                else:
                    for indx in range(hash_array.shape[0]):
                        if not(has_phiB or has_psiB):
                            self.hashes_store["stub"][hash_array["keyStub"][indx]].append([aaPair, indx])
                        elif has_phiB and has_psiB:
                            self.hashes_store["phipsi"][hash_array["keyTor"][indx]].append([aaPair,indx])
                        elif has_phiB:
                            self.hashes_store["phi"][hash_array["keyTor"][indx]].append([aaPair,indx])
                        elif has_psiB:
                            self.hashes_store["psi"][hash_array["keyTor"][indx]].append([aaPair, indx])
                        else:
                            print("There is a mistake, this should never happen!")
                            assert()
    
                #Set number of chis in the table (they are sequencial)
                self.num_chis[aaPair]=[0,0]
                if (hash_array["chisA"].size>0):
                    self.num_chis[aaPair][0]=hash_array["chisA"].shape[1]
                if (hash_array["chisB"].size>0):
                    self.num_chis[aaPair][1]=hash_array["chisB"].shape[1]
                    
                #DEBUG
                #break #DEBUG
                #END DEBUG
        #Sanitty check
        ####assert(len(self.hashes_store["stub"])>0)
        print("Seting important torsions. PLEASE ELIMINATE ME!!!!!! I am not needed")
        for iaapair in self.historic_data_dic:
            self.historic_data_dic[iaapair]['scbb_torsions_A']=np.unique(self.historic_data_dic[iaapair]['scbb_torsions_A'])
            print("  >Torsions_A:", iaapair, self.historic_data_dic[iaapair]['scbb_torsions_A'])
            self.historic_data_dic[iaapair]['scbb_torsions_B']=np.unique(self.historic_data_dic[iaapair]['scbb_torsions_B'])
            print("   >Torsions_B:", iaapair, self.historic_data_dic[iaapair]['scbb_torsions_B'])
            self.dof_resA = self.historic_data_dic[iaapair]['scbb_torsions_A']
            self.dof_resB = self.historic_data_dic[iaapair]['scbb_torsions_B']
            
        #Retrieve and validate hash_function values:
        print("Validating hashing options from database")
        self.stored_options={}
        for ioption in ['hash_function_tor_resl',
                           'hash_function_cart_resl',
                           'hash_function_ori_resl',
                           'hash_function_cart_bound']:
            self.stored_options[ioption]=[]
            for iaapair in self.historic_data_dic:
                for jelement in self.historic_data_dic[iaapair][ioption]:
                    self.stored_options[ioption].append(jelement)
            self.stored_options[ioption]=np.unique(self.stored_options[ioption])
            
           
            
            assert(len(self.stored_options[ioption])==1)
            self.stored_options[ioption]=self.stored_options[ioption][0]
            
        print("Creating score_pair_pose")
        self.scoring_pair_pose=pyrosetta.pose_from_sequence("A")
        tmp_scoring_pair_poseB=pyrosetta.pose_from_sequence("A")
        self.scoring_pair_pose.append_pose_by_jump(tmp_scoring_pair_poseB,1)
        
        #Automatic energy thresholds:
        print("Setting energy and Nhbonds thresholds")
        for iaapair in self.historic_data_dic:
            self.hbondenergy_threshold[iaapair]=(np.average([item for sublist in self.historic_data_dic[iaapair]['historic_pairHBondsEnergy'] for item in sublist])+
                                            (times_std_dev_threshold*np.std([item for sublist in self.historic_data_dic[iaapair]['historic_pairHBondsEnergy'] for item in sublist])))
            self.hbondenergy_threshold[iaapair]=min(max_pair_energy, self.hbondenergy_threshold[iaapair])
            
            self.nhbonds_threshold[iaapair]=(np.average([item for sublist in self.historic_data_dic[iaapair]['historic_pairNHbonds'] for item in sublist])-
                                            (times_std_dev_threshold*np.std([item for sublist in self.historic_data_dic[iaapair]['historic_pairNHbonds'] for item in sublist])))
            self.nhbonds_threshold[iaapair]=max(min_num_hbonds_pair, self.nhbonds_threshold[iaapair]-nhbond_threshold_nbonus)
        print ("Thresholds E:", self.hbondenergy_threshold)
        print ("Thresholds NHbonds:", self.nhbonds_threshold)
        
        print("Initializing Rosetta")
        #Init scoring function
        #ToDo pass an instance of pyrosetta: DO NOT INIT HERE
        #pyrosetta.init("-beta -mute all")
        self.sfx = self.hb_energy_fxn("beta_nov15")
        
        """
        hash_function=rif.hash.RosettaStubTorsionHash(self.stored_options['hash_function_phi_resl'], 
                                                      self.stored_options['hash_function_cart_resl'], 
                                                      self.stored_options['hash_function_ori_resl'], 
                                                      self.stored_options['hash_function_cart_bound'])
        
        """
        
        print("Initializing Mutators Dictionary")
        for aapair in self.aapairs_poseMutators:
            print("Pair:", aapair, ",to a.a.s:", aapair[0][:3], aapair[1][:3])
            self.aapairs_poseMutators[aapair]=[pyrosetta.rosetta.protocols.simple_moves.MutateResidue(),
                                               pyrosetta.rosetta.protocols.simple_moves.MutateResidue()]
            
            if ((aapair[0][:3] != "GLY") or do_use_gly):
                self.aapairs_poseMutators[aapair][0].set_res_name(aapair[0][:3])
            else:
                self.aapairs_poseMutators[aapair][0]=self.do_nothing_to_pose()
                self.num_chis[aapair][0]=0
            if ((aapair[1][:3] != "GLY") or do_use_gly):
                self.aapairs_poseMutators[aapair][1].set_res_name(aapair[1][:3])
            else:
                self.aapairs_poseMutators[aapair][1]=self.do_nothing_to_pose()
                self.num_chis[aapair][1]=0
        print("Initializing transform hashes with options:", self.stored_options)
        super(use_hash_universal, self).__init__(self.stored_options['hash_function_tor_resl'],
                                        self.stored_options['hash_function_cart_resl'], 
                                        self.stored_options['hash_function_ori_resl'], 
                                        self.stored_options['hash_function_cart_bound'])
        
        print("All init Done. Hashes in the database:", len(self.hashes_store))

        
    #set up score function
    def hb_energy_fxn(self,
                      scorefxn_name="beta_nov15"):
        sf = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(scorefxn_name)
        sf.set_weight(pyrosetta.rosetta.core.scoring.fa_dun, 0.5) #0.7
        sf.set_weight(pyrosetta.rosetta.core.scoring.hbond_bb_sc, 2.0) #1.0
        sf.set_weight(pyrosetta.rosetta.core.scoring.hbond_sc, 2.0) #1.0
        return sf
        
    #two options when using hash:  
    ## 1) count number of hits (bidentate hbs) and replace residue types in pose for each hit and
    ## 2) return full info for each hit (all rots and relevant chis).            
    def find_resPair_match_in_pose(self,
                     prot,
                     contact_list_A=None,
                     contact_list_B=None,
                     torsion_space=["stub"],
                     min_dist_check=4.0,
                     max_dist_check=14.0,
                     check_clashes_in_pose=True,
                     insertion_energy_threshold=100.0): #Potentially should add reversibility for "phipsi"
    
        if contact_list_B is None:
            contact_list_B = contact_list_A
        init_pose = prot.clone()
        init_score = self.sfx(init_pose)

        result_list=[]
        for torsions in torsion_space:
            for iresN in contact_list_A:
                frameA = self.get_frame_from_res(prot.residue(iresN))
                for jres in contact_list_B: #range(1,pept.size()+1):
                    # NOTE(onalant): start task delegation here
                    ab_dist=(prot.residue(jres).xyz("CA")-prot.residue(iresN).xyz("CA")).norm()
                    if((ab_dist>min_dist_check) and 
                       (ab_dist<max_dist_check)):
                        #ToDo: coalese_phi_psi hashes in BB-only and search on phi and psi afterwards

                        k=self.hash_functions[torsions](frameA, prot, jres)
                        ##print("Looking for:", iresN, jres, k_phi, k_psi)
                        if (k in self.hashes_store[torsions]):
                            ##print("WHAT_A:?", torsions, k)
                            result_vector= self.place_residues_and_chis(prot,
                                                                        k,
                                                                        iresN,
                                                                        jres,
                                                                        torsions,
                                                                        check_clashes_in_pose=check_clashes_in_pose,
                                                                        insertion_energy_threshold=insertion_energy_threshold)

                            for kresult in result_vector:
                                # NOTE(onalant): score and check for deltas in all energy terms
                                # BEGIN: energy calculation of replaced residue pair

                                tmp_pose = prot.clone()
                                tmp_pose.replace_residue(iresN, kresult[-2], False)
                                tmp_pose.replace_residue(jres, kresult[-1], False)

                                tmp_score = self.sfx(tmp_pose)
                                delta_score = tmp_score - init_score

                                # END

                                if (delta_score < insertion_energy_threshold):
                                    result_list.append([torsions,delta_score]+kresult)

                                
                                
                
                        frameB = self.get_frame_from_res(prot.residue(jres))
                        k=self.hash_functions[torsions](frameB, prot, iresN)
                        ##print("Looking for:", iresN, jres, k_phi, k_psi)
                        if (k in self.hashes_store[torsions]):
                            ##print("WHAT_B:?", torsions, k)
                            result_vector= self.place_residues_and_chis(prot,
                                                                           k,
                                                                           jres,
                                                                           iresN,
                                                                           torsions,
                                                                           check_clashes_in_pose=check_clashes_in_pose,
                                                                           insertion_energy_threshold=insertion_energy_threshold)
                        
                            for kresult in result_vector:
                                # NOTE(onalant): score and check for deltas in all energy terms
                                # BEGIN: energy calculation of replaced residue pair
                        
                                tmp_pose = prot.clone()
                                tmp_pose.replace_residue(jres, kresult[-2], False)
                                tmp_pose.replace_residue(iresN, kresult[-1], False)
                        
                                tmp_score = self.sfx(tmp_pose)
                                delta_score = tmp_score - init_score
                        
                                # END
                        
                                if (delta_score < insertion_energy_threshold):
                                    result_list.append([torsions,delta_score]+kresult)
                                    
        return result_list

    def find_resPair_match_between_poses(self,
                     prot,
                     pept,
                     contact_list_prot,
                     contact_list_pept,
                     torsion_space=["stub"],
                     min_dist_check=4.0,
                     max_dist_check=14.0,
                     check_clashes_in_pose=True,
                     insertion_energy_threshold=50.0):
        cumulative = prot.clone()
        num_residue = cumulative.total_residue();
        cumulative.append_pose_by_jump(pept, num_residue)

        contact_list_pept = list(map(lambda x: x + num_residue, contact_list_pept))

        return self.find_resPair_match_in_pose(cumulative,
                                           contact_list_prot,
                                           contact_list_pept,
                                           torsion_space,
                                           min_dist_check,
                                           max_dist_check,
                                           check_clashes_in_pose=check_clashes_in_pose,
                                           insertion_energy_threshold=insertion_energy_threshold)
        
    def extract_aa_pair_from_pose_and_scoreWhbods(self,
                                              target_pose=None,
                                              indexA=1,
                                              indexB=2,
                                              sfx=None):

        self.scoring_pair_pose.replace_residue( seqpos=1, 
                                          new_rsd_in=target_pose.residue(indexA), 
                                          orient_backbone=False );
        self.scoring_pair_pose.replace_residue( seqpos=2, 
                                          new_rsd_in=target_pose.residue(indexB), 
                                          orient_backbone=False );


        #####HBONDS####
        #Calculate num of Hbonds in the pair
        self.scoring_pair_pose.update_residue_neighbors()
        hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
        hbond_set.setup_for_residue_pair_energies(self.scoring_pair_pose, 
                                                  False, 
                                                  False);

        #Count unique hbonds and also bb_bb:
        bb_bb_counter=0
        unique_hbond_atoms_A=set()
        unique_hbond_atoms_B=set()        
        for ibond in range(1,hbond_set.nhbonds()+1):
            #print (hbond_set.hbond(ibond).acc_res(), hbond_set.hbond(ibond).acc_atm())
            unique_hbond_atoms_A.add((hbond_set.hbond(ibond).acc_res(), hbond_set.hbond(ibond).acc_atm()))
            unique_hbond_atoms_B.add((hbond_set.hbond(ibond).don_res(), hbond_set.hbond(ibond).don_hatm()))
            if (hbond_set.hbond(ibond).don_hatm_is_backbone() and 
                hbond_set.hbond(ibond).acc_atm_is_backbone()):
                bb_bb_counter+=1

        n_unique_h_bonds=min(len(unique_hbond_atoms_A), len(unique_hbond_atoms_B))
        return([sfx(self.scoring_pair_pose), 
                n_unique_h_bonds,
                n_unique_h_bonds-bb_bb_counter])
###

    #Finds optimum result and places in the input pose, returns in a vector
    def place_residues_and_chis(self,
                                pose,
                                key,
                                jresA,
                                jresB,
                                torsions,
                                check_clashes_in_pose=True,
                                insertion_energy_threshold=50.0):
        results_arr=[]
        initial_combined_pose_score=False
        if check_clashes_in_pose:
            initial_combined_pose_score=self.sfx(pose)
            
        #Reduce the complexity of the result
        reduced_aapairs_chisAB=collections.defaultdict(set)
        for jaapair,jhashndx in self.hashes_store[torsions][key]:
            reduced_aapairs_chisAB[jaapair].add((tuple(self.hash_array_dic_chisA[jaapair][jhashndx]),
                                                 tuple(self.hash_array_dic_chisB[jaapair][jhashndx])))
        
        ##print(reduced_aapairs_chisAB)
        for jaapair in reduced_aapairs_chisAB:
            for chisA,chisB in  reduced_aapairs_chisAB[jaapair]:
                ##print (chisA," : ",chisB) #Debug
                tmp_pose = pose.clone()

                self.aapairs_poseMutators[jaapair][0].set_target(jresA)
                self.aapairs_poseMutators[jaapair][1].set_target(jresB)
                self.aapairs_poseMutators[jaapair][0].apply(tmp_pose)
                self.aapairs_poseMutators[jaapair][1].apply(tmp_pose)

                """
                tmp_poseA.pdb_info().add_reslabel(jresA,"mDBnet_%s"%jaapair[0])
                tmp_poseA.pdb_info().add_reslabel(jresA,"mDBnet")
                tmp_poseB.pdb_info().add_reslabel(jresB,"mDBnet_%s"%jaapair[1])
                tmp_poseB.pdb_info().add_reslabel(jresB,"mDBnet")
                """

                #best_score_hbonds=[99999.99,-1]
                ##tmp_col_counter=0
                #ToDo:
                #1.Fix the number of deegrees of freedom sence they are dependant on the database (i.e. they might not even be there)
                for ichi in range(self.num_chis[jaapair][0]):#tmp_poseA.residue(jresA).nchi()):
                    tmp_pose[jresA].set_chi(ichi+1, chisA[ichi]*2.0)
                for ichi in range(self.num_chis[jaapair][1]):#tmp_poseB.residue(jresB).nchi()):
                    tmp_pose[jresB].set_chi(ichi+1, chisB[ichi]*2.0)

                tmpscore,nHbonds,nHbonds_noBb=self.extract_aa_pair_from_pose_and_scoreWhbods(
                                                                                    tmp_pose,
                                                                                    indexA=jresA,
                                                                                    indexB=jresB,
                                                                                    sfx=self.sfx)
                #If enough Energy and hbonds
                if ((tmpscore<=self.hbondenergy_threshold[jaapair]) and
                    (nHbonds_noBb>=self.nhbonds_threshold[jaapair])):
                    if check_clashes_in_pose: 
                        delta_score=self.sfx(tmp_pose)-initial_combined_pose_score
                        ##print(delta_score, nHbonds_noBb) #DEBUG
                        if (delta_score<insertion_energy_threshold):
                            results_arr.append([jresA,
                                                jresB,
                                                tmpscore,
                                                nHbonds_noBb,
                                                tmp_pose.residue(jresA),
                                                tmp_pose.residue(jresB)])
                                                #tmp_pose.clone()])
                    else:
                        results_arr.append([jresA,
                                            jresB,
                                            tmpscore,
                                            nHbonds_noBb,
                                            tmp_pose.residue(jresA),
                                            tmp_pose.residue(jresB)])
                                            #tmp_pose.clone()])

        return results_arr
