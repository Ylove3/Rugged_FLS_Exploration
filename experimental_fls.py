import os
import collections
import itertools
import re
import numpy as np
from scipy import sparse
import scipy.stats as ss
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tables
import h5py
from scipy.integrate import trapz
import matplotlib.animation as animation
from pylab import *


class Exp_FLS_Plugin(object):
    '''
    A class of plugins that, given the filename of the data, creates the canonized pandas df of 
    columns=['Num', 'Fit','Seq',...] where 
    'Num' is an integert - number of mutation, 
    'Fit' is a float, 
    'Seq' is a string.
    
    '''
    def __init__(self, filename, fls_builder, states = ['A', 'C', 'T', 'G'], N = 72):
        
        '''
        The 2 most important inputs here are the filename and the fls_builder function. 
        '''
        
        self.states = states
        self.gene_length = N
        self.filename = filename
        self.build_func = fls_builder
        self.data = fls_builder(self.filename)
        
def tRNA_fls_builder(filename):
    
    data = pd.read_csv(filename, sep='\t')
    
    return data
    
def gfp_fls_builder(filename):

    '''
    loads the data and performs the following:
    1. clean it from Nans
    2. adding number of nt mutations and aa mutations
    3. renaming columns
    4. adding sequence column
    5. reordering according to nt mutations
    '''
    
    
    data = pd.read_csv(filename, sep='\t')
    data.fillna(0, inplace = True)
    
    c = lambda s: s if s==0 else (s.count(':')+1)
    data['Num'] = data['nMutations'].apply(c)
    data['aaNum'] = data['aaMutations'].apply(c)
    data.rename(columns={'medianBrightness': 'Fit'}, inplace=True)
    
    def wt_seq_gen(mut_str):
    
        wt = 'XXAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGTCATACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACXXXXXXGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCACGGCATGGACGAGCTGTAC'
        list_wt=list(wt)

        if mut_str==0:
            return wt
        p = re.compile('\w[0-9]+\w')
        all_matches = p.findall(mut_str)
        l = [(i[0], i[1:-1], i[-1]) for i in all_matches]
        for a in l:
            assert list_wt[int(a[1])-1] == a[0]
            list_wt[int(a[1])-1]=a[2]
        return (''.join(list_wt)) # Here are loosing the 2 X's and we now get a shift of (-3) from the original nMutations field
    
    data['Seq'] = data['nMutations'].apply(wt_seq_gen)
    
    data.sort_values(by=['Num', 'aaNum'], inplace = True)
    
    return data.reset_index(drop = True)
       
class Exp_Pipline_Analyzer(object):
    
    '''
    This class includes a set of tools that will help one to analyze different experimental FLS's. 
    most relevant:
    
    1. firsht description of the data (fitness and mutation distributions)
    2. hamming distance calculator
    3. stability analysis of a given point
    4. trajectories calculator of depth = 2 (either strictly increasing/decreasing or just all possible trajectories)
    5. peaks calculation
    
    In order to initialize it you only need to pass canonized data from Exp_FLS_Plugin
    
    '''
    
    def __init__(self, data, mode):
        
        self.data = data
        self.mode = mode
        self.stacked_diff_matrix = None
        self.stacked_diff_matrix_filename = None
        self.access_root = '/stacked_diff_matrix'
        if self.mode=='exp':
            self.seq2numbers()
        
        
    def simple_data_descriptor(self):
        
        plt.subplot(3,1,1)
        print("General statistics of the data:")
        print(self.data.describe())
        
        plt.subplot(3,1,2)
        self.data['Fit'].hist()
        plt.title("histogram of the fitness values")
                
        plt.subplot(3,1,3)
        self.data['Num'].hist()
        plt.title("histogram of the mutations")
    
    def seq2numbers(self, method = 'nt', alphabet = None ):    
        '''
        Future plans - develop it to general alhpabet and to differnet methods (such as proteins, genes, whatever)
        '''
        if method == 'nt':
            nt_dict = {'X':'0','A':'0', 'T':'1', 'C':'2','G':'3'}

        to_genotype = lambda x: nt_dict[x]

        self.data['Genotype'] = self.data['Seq'].apply(
            lambda seq: str.join('', map(to_genotype, list(seq))))        
    
    def genotype2vector(self, genotype):
        
        return [int(i) for i in list(genotype)]
    
    def all_genotypes_matrix(self, amount = -1):
        
        '''
        Returns all the genptypes as column vectors stacked inside a big matrix, ready for hamming distance calculation
        '''
        if amount < 0:
            all_genotypes = np.array(list(self.data['Genotype'].apply(self.genotype2vector))).T
        else:
            all_genotypes = np.array(list(self.data['Genotype'].head(amount).apply(self.genotype2vector))).T
            
        return all_genotypes
    
    def hamming_calculator(self, save_filename, min_index = 0, max_index = -1, costum_index = []):
    
        
        '''
        Goal: calculating all the hamming distances of the desired amount of variants with all the others in the data
        
        Input: 
        
        max_index - int, the maximal variant index in self.data df for which the function will calculate. 
                    the default - it will be calculated for the entire corpus
                    
        save_filename - path string, the filename that the resulting matrix will be saved to
        
        Output: 
        
        stacked_diff_matrix - numpy array, the size of max_index X amount_of_all_genotypes       
        '''
        
        all_genotypes = self.all_genotypes_matrix()
        
        def _hamming_matrice_creator(row_num, all_genotypes):
            
            # This function creates a diagonal (-1) matrice with only 1's in the current seq. row number we are looking at
                
            N_variants = all_genotypes.shape[1]   
            dif_matrix = sparse.diags([-1]*N_variants) + sparse.csr_matrix(([1]*N_variants, ([row_num]*N_variants,list(range(N_variants)))), shape=(N_variants,N_variants))
            return dif_matrix
        
        N_variants = all_genotypes.shape[1]
        if max_index < 0:
            max_index = N_variants
        
        if len(costum_index)!=0:
            
#             print("Note- You are about to calculate a {} X {} dot a {} X {} matrix multiplication, {} times...".format(all_genotypes.shape[0], all_genotypes.shape[1], all_genotypes.shape[1], all_genotypes.shape[1], max_index))
#             answer = input('Would you like to continue? y/n:')
#             if (answer == 'N') | (answer == 'n'):
#                 return -1

            stacked_diff_matrix = np.empty([0, N_variants])
            #print(stacked_diff_matrix.shape)

            for j, i in enumerate(costum_index):

                if j%200 == 0:
                    print("{} multiplications have been calculated so far".format(j))
                hamming_matrice = _hamming_matrice_creator(i, all_genotypes) # creates a 65K X 65K-1 matrice (we won't have the hamming of this seq with itself)
                hamming_matrice = sparse.csr_matrix.dot(all_genotypes, hamming_matrice) # resulting a 72 X 65K -1 differences 
                hamm_dist = np.count_nonzero(hamming_matrice, axis = 0) # resulting a 65K-1 vector with the amount of differences
                stacked_diff_matrix = np.vstack((stacked_diff_matrix, hamm_dist))

            np.save(save_filename, stacked_diff_matrix)
            self.stacked_diff_matrix = stacked_diff_matrix

            return stacked_diff_matrix
            
        
        print("Note- You are about to calculate a {} X {} dot a {} X {} matrix multiplication, {} times...".format(all_genotypes.shape[0], all_genotypes.shape[1], all_genotypes.shape[1], all_genotypes.shape[1], max_index-min_index))
                                                                                                                                
#         answer = input('Would you like to continue? y/n:')
#         if (answer == 'N') | (answer == 'n'):
#             return -1
            
        stacked_diff_matrix = np.empty([0, N_variants])
        #print(stacked_diff_matrix.shape)

        for i in range(min_index, max_index):

            if i%2000 == 0:
                print("{} multiplications have been calculated so far".format(i))
                np.save(save_filename, stacked_diff_matrix)
                self.stacked_diff_matrix = stacked_diff_matrix
                
            hamming_matrice = _hamming_matrice_creator(i, all_genotypes) # creates a 65K X 65K-1 matrice (we won't have the hamming of this seq with itself)
            hamming_matrice = sparse.csr_matrix.dot(all_genotypes, hamming_matrice) # resulting a 72 X 65K -1 differences 
            hamm_dist = np.count_nonzero(hamming_matrice, axis = 0) # resulting a 65K-1 vector with the amount of differences
            stacked_diff_matrix = np.vstack((stacked_diff_matrix, hamm_dist))
        
        np.save(save_filename, stacked_diff_matrix)
        self.stacked_diff_matrix = stacked_diff_matrix
        
        return stacked_diff_matrix
    
    def robust_hamming_calculator(self, save_filename, min_index = 0, max_index = -1, costum_index = []):
        
        '''
        Goal: calculating all the hamming distances of the desired amount of variants with all the others in the data
        
        Input: 
        
        max_index - int, the maximal variant index in self.data df for which the function will calculate. 
                    the default - it will be calculated for the entire corpus
                    
        save_filename - path string, the filename that the resulting matrix will be saved to
        
        Output: 
        
        stacked_diff_matrix - numpy array, the size of max_index X amount_of_all_genotypes       
        '''
        
        all_genotypes = self.all_genotypes_matrix()
        
        def _hamming_matrice_creator(row_num, all_genotypes):
            
            # This function creates a diagonal (-1) matrice with only 1's in the current seq. row number we are looking at
                
            N_variants = all_genotypes.shape[1]   
            dif_matrix = sparse.diags([-1]*N_variants) + sparse.csr_matrix(([1]*N_variants, ([row_num]*N_variants,list(range(N_variants)))), shape=(N_variants,N_variants))
            return dif_matrix
        
        N_variants = all_genotypes.shape[1]
        if max_index < 0:
            max_index = N_variants
        
        if len(costum_index)!=0:
            
#             print("Note- You are about to calculate a {} X {} dot a {} X {} matrix multiplication, {} times...".format(all_genotypes.shape[0], all_genotypes.shape[1], all_genotypes.shape[1], all_genotypes.shape[1], max_index))
#             answer = input('Would you like to continue? y/n:')
#             if (answer == 'N') | (answer == 'n'):
#                 return -1

            stacked_diff_matrix = np.empty([0, N_variants])
            #print(stacked_diff_matrix.shape)

            for j, i in enumerate(costum_index):

                if j%200 == 0:
                    print("{} multiplications have been calculated so far".format(j))
                hamming_matrice = _hamming_matrice_creator(i, all_genotypes) # creates a 65K X 65K-1 matrice (we won't have the hamming of this seq with itself)
                hamming_matrice = sparse.csr_matrix.dot(all_genotypes, hamming_matrice) # resulting a 72 X 65K -1 differences 
                hamm_dist = np.count_nonzero(hamming_matrice, axis = 0) # resulting a 65K-1 vector with the amount of differences
                stacked_diff_matrix = np.vstack((stacked_diff_matrix, hamm_dist))

            np.save(save_filename, stacked_diff_matrix)
            self.stacked_diff_matrix = stacked_diff_matrix

            return stacked_diff_matrix
            
        
        print("Note- You are about to calculate a {} X {} dot a {} X {} matrix multiplication, {} times...".format(all_genotypes.shape[0], all_genotypes.shape[1], all_genotypes.shape[1], all_genotypes.shape[1], max_index-min_index))
                                                                                                                                
#         answer = input('Would you like to continue? y/n:')
#         if (answer == 'N') | (answer == 'n'):
#             return -1
            
        stacked_diff_matrix = np.empty([0, N_variants])
        #print(stacked_diff_matrix.shape)

        for i in range(min_index, max_index):

            if i%2000 == 0:
                print("{} multiplications have been calculated so far".format(i))
                np.save(save_filename, stacked_diff_matrix)
                self.stacked_diff_matrix = stacked_diff_matrix
                
            hamming_matrice = _hamming_matrice_creator(i, all_genotypes) # creates a 65K X 65K-1 matrice (we won't have the hamming of this seq with itself)
            hamming_matrice = sparse.csr_matrix.dot(all_genotypes, hamming_matrice) # resulting a 72 X 65K -1 differences 
            hamm_dist = np.count_nonzero(hamming_matrice, axis = 0) # resulting a 65K-1 vector with the amount of differences
            stacked_diff_matrix = np.vstack((stacked_diff_matrix, hamm_dist))
        
        np.save(save_filename, stacked_diff_matrix)
        self.stacked_diff_matrix = stacked_diff_matrix
        
        return stacked_diff_matrix
    
    def entire_hamming_calculator(self, save_filename):
        
        '''
        Goal: calculating all the hamming distances of the desired amount of variants with all the others in the data.
        Since usually we are talking on pretty big matrices, we will use hdf / h5 file format to access the data. 
        
        Input: 
        
        save_filename - path string, the filename that the resulting matrix will be saved to
        
        Output: 
        
        stacked_diff_matrix_filename - filename of the h5-format file that the matrix was save to
        access - access root to the matrix
        
        '''
                
        def _hamming_matrice_creator(row_num, all_genotypes):
            
            # This function creates a diagonal (-1) matrice with only 1's in the current seq. row number we are looking at
                
            N_variants = all_genotypes.shape[1]   
            dif_matrix = sparse.diags([-1]*N_variants) + sparse.csr_matrix(([1]*N_variants, ([row_num]*N_variants,list(range(N_variants)))), shape=(N_variants,N_variants))
            return dif_matrix
            
        def _hamm_matrix_from_min_max(min_index, max_index, N_variants, all_genotypes):
            
            # This function is automaton that calculates hamming matrix from (min,max) ranges. Lets assume that max-min = p
            
            stacked_diff_matrix = np.zeros([0, N_variants])

            for i in range(min_index, max_index):

                if i%2000 == 0:
                    print("{} multiplications have been calculated so far".format(i))
                    
                hamming_matrice = _hamming_matrice_creator(i, all_genotypes) # creates a p X p-1 matrice (we won't have the hamming of this seq with itself)
                hamming_matrice = sparse.csr_matrix.dot(all_genotypes, hamming_matrice) # resulting a all_genotypes.shape[0] (== which is the length of the genome) X p -1 differences 
                hamm_dist = np.count_nonzero(hamming_matrice, axis = 0) # resulting a p-1 vector with the amount of differences
                stacked_diff_matrix = np.vstack((stacked_diff_matrix, hamm_dist))
            
            return stacked_diff_matrix            

        all_genotypes = self.all_genotypes_matrix()
        N_variants = all_genotypes.shape[1]
        
        chunk_size = 6000
        # creating the hdf file to write to
        f_name = save_filename+'.hdf'
        f = tables.open_file(f_name, 'w')
        atom = tables.Atom.from_dtype(np.dtype(float64))
        filters = tables.Filters(complib='blosc', complevel=5)
        ds = f.create_carray(f.root, 'stacked_diff_matrix', atom, (N_variants, N_variants) , filters=filters)
        # save w/o compressive filter
        #ds = f.createCArray(f.root, 'all_data', atom, all_data.shape)
        f.close()
        
        chunks = list(range(0,N_variants,chunk_size))
        if (N_variants-1)%chunk_size == 0:
            chunks.remove(N_variants)
        
        f = h5py.File(f_name, mode='r+')        
        for chunk in chunks:
            
            current_start_index = chunk
            if chunk == chunks[-1]:
                current_end_index = N_variants
            else:
                current_end_index = current_start_index + chunk_size
                
            current_mat = _hamm_matrix_from_min_max(current_start_index, current_end_index, N_variants, all_genotypes)          
            f['/stacked_diff_matrix'][current_start_index:current_end_index,:] = current_mat
            
        f.close()
        access_root = '/stacked_diff_matrix' 
        self.stacked_diff_matrix_filename = f_name
        self.access_root = access_root
        
        return f_name, access_root
    
    def stability_fields_calc(self, r = 1, fit_column = 'Fit'):
        
        '''
        This function will add the number of 1-neighbors each strain has inside the data, as long as 
        stacked_diff_matrix has a row with the hamming distances for this variant.
        
        further more it calculates the 1-neighbors-gradient (which equals = center_fitness - current_fitness) mean, 
        median and std + the same for the absolute values of each of the gradient vectors. 
        This serves as a stability measure.
        
        Output - a revised self.data table with all the new fields.
        '''
        
        r_str = str(r)
        
        self.data[r_str+'-neig'] = np.nan
        self.data[r_str+'-grad-mean-'+fit_column] = np.nan
        self.data[r_str+'-grad-median-'+fit_column] = np.nan
        self.data[r_str+'-grad-std-'+fit_column] = np.nan
        self.data[r_str+'-grad-std-abs-'+fit_column] = np.nan
        self.data[r_str+'-abs-grad-mean-'+fit_column] = np.nan
        self.data[r_str+'-abs-grad-median-'+fit_column] = np.nan
        
        continues = 0
        
        f = h5py.File(self.stacked_diff_matrix_filename, 'r')
        n_runs = f[self.access_root].shape[0]
        
        for i in list(range(n_runs)):

            if i%2000==0:
                print("{} variants have been treated so far".format(i))
    
            self.data.loc[i, r_str+'-neig'] = np.count_nonzero(f[self.access_root][i,:]==r)
        
            if self.data.loc[i, r_str+'-neig'] == 0:
                #print(" in index {}, there are no 1-neighbors, continuing".format(i))
                continues = continues + 1
                continue
            center_fitness = self.data.loc[i, fit_column]
            fitness_vector_of_r_neig = self.data.loc[np.where(f[self.access_root][i,:]==r)][fit_column]
            
            self.data.loc[i,r_str+'-grad-mean-' + fit_column] = np.mean(center_fitness-fitness_vector_of_r_neig)
            self.data.loc[i,r_str+'-grad-median-' + fit_column] = np.median(center_fitness-fitness_vector_of_r_neig)
            self.data.loc[i,r_str+'-grad-std-' + fit_column] = np.std(center_fitness-fitness_vector_of_r_neig)
            self.data.loc[i,r_str+'-grad-std-abs-' + fit_column] = np.std(np.abs(center_fitness-fitness_vector_of_r_neig))
            self.data.loc[i,r_str+'-abs-grad-mean-' + fit_column] = np.mean(np.abs(center_fitness-fitness_vector_of_r_neig))
            self.data.loc[i,r_str+'-abs-grad-median-' + fit_column] = np.median(np.abs(center_fitness-fitness_vector_of_r_neig))
            
        f.close()
        print("Finished, #continues was {} out of {}. You should take into acount that they are Nans".format(continues, n_runs))
    
    def trajectories_calc(self, root = 0, delta = 0, max_index=-1, method = 'decreasing', fitness_column = 'Fit'):
    
    
        '''
        Input:

        root - the INDEX-NUMBER in self.data of the genotype that we want to create trajectories from. Usually it will be 
            the WT which normally will reside in the first row (index==0)

        delta = gives you some freedom to choose what are 2 equifitness genotypes. e.g. for root it will go according to this:
            fitness(neigbour) <= fitness(root) + delta. In the 'decreasing' method we are only looking for delta-decreasing trajectories

        max_index - since stacked_diff_matrix is hard to compute we usually don't have it for every index.. 
            so here is an option to control up to what index we would like to calculate the trajectories. 
            the default is this: max_index = stacked_diff_matrix.shape[0]
            
        method - There are a few methods that can be considered here: 'decreasing' (default), 'increasing' trajectories and 
                (delta) 'neutral' networks.

        Returns:

        trajectories - pandas df that holds all the delta-decreasing/increasing/neutral trajectories and their fitnesses and their fitnesses 
            differences (only for the non-neutral methods)

        '''

        TOL = delta

        # Creates a product easy to work with - but it's slower than the recursive function.

        trajectories = pd.DataFrame(columns=['father', 'father_fitness', 'child', 'child_fitness', 
                                   'grandchild', 'grandchild_fitness', 'g_grandchild', 
                                   'g_grandchild_fitness'])

        dict2append = {'father' : 0, 'father_fitness' : 0 , 'child': 0, 'child_fitness': 0, 
                       'grandchild': 0, 'grandchild_fitness': 0, 'g_grandchild': 0, 
                       'g_grandchild_fitness' : 0}

        dict2append['father'] = root
        dict2append['father_fitness'] = self.data.loc[root, fitness_column]
        
        f = h5py.File(self.stacked_diff_matrix_filename, 'r')
        
        if max_index<0:
            max_index = f[self.access_root].shape[0]
        
        root_neig = np.where(f[self.access_root][root] == 1)[0] # The indexes in data of the 1-neig of the root-index
        neig_data = self.data.loc[root_neig]
        if method == 'decreasing':
            root_lower_fit = list(neig_data[neig_data[fitness_column] <= (self.data.loc[root,fitness_column] + TOL)].index.values)
        elif method == 'increasing':
            root_lower_fit = list(neig_data[neig_data[fitness_column] >= (self.data.loc[root,fitness_column] - TOL)].index.values)
        elif method == 'neutral':
            root_lower_fit = list(neig_data[(neig_data[fitness_column] <= (self.data.loc[root,fitness_column] + TOL)) & 
                                            (neig_data[fitness_column] >= (self.data.loc[root,fitness_column] - TOL))].index.values)
        else:
            print('unknown method named {}'.format(method))
            return -1
        # Truncating lower_fit to be only with indexes up to 24K where we have data
        root_lower_fit = [x for x in root_lower_fit if x < max_index]

        counter = 0

        for child in root_lower_fit:

            dict2append['child'] = child
            dict2append['child_fitness'] = self.data.loc[child, fitness_column]

            child_neig = np.where(f[self.access_root][child]==1)[0]
            neig_data = self.data.loc[child_neig]
            
            if method == 'decreasing':
                child_lower_fit = list(neig_data[neig_data[fitness_column] <= (self.data.loc[child,fitness_column] + TOL)].index.values)
            elif method == 'increasing':
                child_lower_fit = list(neig_data[neig_data[fitness_column] >= (self.data.loc[child,fitness_column] - TOL)].index.values)
            elif method == 'neutral':
                child_lower_fit = list(neig_data[(neig_data[fitness_column] <= (self.data.loc[child,fitness_column] + TOL)) & 
                                      (neig_data[fitness_column] >= (self.data.loc[child,fitness_column] - TOL))].index.values)
            
            child_lower_fit = [x for x in child_lower_fit if x < max_index]

            for grandchild in child_lower_fit:

                dict2append['grandchild'] = grandchild  # This is last father in the recursion function
                dict2append['grandchild_fitness'] = self.data.loc[grandchild, fitness_column]

                grandchild_neig = np.where(f[self.access_root][grandchild]==1)[0]
                neig_data = self.data.loc[grandchild_neig]
                
                if method == 'decreasing':
                    grandchild_lower_fit = list(neig_data[neig_data[fitness_column] <= (self.data.loc[grandchild,fitness_column] + TOL)].index.values)
                elif method == 'increasing':
                    grandchild_lower_fit = list(neig_data[neig_data[fitness_column] >= (self.data.loc[grandchild,fitness_column] - TOL)].index.values)
                elif method == 'neutral':
                    grandchild_lower_fit = list(neig_data[(neig_data[fitness_column] <= (self.data.loc[grandchild,fitness_column] + TOL)) & 
                                          (neig_data[fitness_column] >= (self.data.loc[grandchild,fitness_column] - TOL))].index.values)
                
                grandchild_lower_fit = [x for x in grandchild_lower_fit if x < max_index]

                for g_grandchild in grandchild_lower_fit:

                    dict2append['g_grandchild'] = g_grandchild  # These are the leaves of the recursion function
                    dict2append['g_grandchild_fitness'] = self.data.loc[g_grandchild, fitness_column]

                    trajectories = trajectories.append(dict2append, ignore_index=True) # .append here works different than lists
                    counter = counter + 1
                    if counter%5000==0:
                        print("{} trajectories have been calculated so far".format(counter))
                        
        if method != 'neutral':
            
            trajectories['father_child_grad'] = trajectories['father_fitness']-trajectories['child_fitness']
            trajectories['child_grandchild_grad'] = trajectories['child_fitness']-trajectories['grandchild_fitness']
            trajectories['grandchild_ggrandchild_grad'] = trajectories['grandchild_fitness'] - trajectories['g_grandchild_fitness']
            trajectories['total_descent'] = trajectories['father_child_grad'] + trajectories['child_grandchild_grad'] + trajectories['grandchild_ggrandchild_grad'] 
        
        f.close()
        
        return trajectories     
       
    def peaks_valleys_calc(self, delta = 0, lower_bound = 0.5):
        
        '''
        
        Goal: This function will calculate all the peaks and valleys (when compared to their 1-neighbors)
               in self.data. For now works only for nt sequences.
        
        Input:
        
        
        delta - again, an equifitness measure: does being bigger by 0.00001 
                than your neighbors makes you a peak? 
                Can also be seen in another way as delta-peaks = checks if there are peaks that are bigger than all of their
                soroundings by more than delta.. (the same goes with valleys only more then delta smaller)
                
        lower_bound - In a lot of experiments they truncate the fitness to reduce noise. 
                      When calculating peaks and valleys we might want to avoid them - this is
                      the threshold of avoidance. default is for tRNA FLS experiment. 
        
        Return: np arrays of peaks and valleys with their fitnesses
        
        '''
        
        fitness = collections.defaultdict(float) # Here we create a dict that will return a float 0.0 if the key you requested doesnt exist
        for i, row in self.data.iterrows():
            fitness[row['Genotype']] = row['Fit']

        
        genotypes = tuple(fitness.keys())
        
        
        def _possible_mut_per_position(genotype, position):
            #Works only for nt sequences!
            current_mut = list(range(4))
            current_mut.pop(int(list(genotype)[position])) # these are all the possible mutations to this specific genome in this specific position
            possible_mutations = []
            for t in current_mut:
                h = list(genotype)
                h[position]=str(t)
                possible_mutations.append(str.join('',h))
            return possible_mutations

        peaks = list(genotypes)
        
        print("strating by calculating the peaks")
        
        for r, g in enumerate(genotypes):
            
            if r%2000==0:
                print("so far went over {} genotypes". format(r))                    
            
            break_flag = 0
            if fitness[g]<=lower_bound:
                peaks.pop(peaks.index(g))
            else:
                for i in range(len(g)):
                    for j in _possible_mut_per_position(g, i):
                        if fitness[j] > fitness[g] - delta: # Takes out any non-qualifiers to be delta-peak 
                            peaks.pop(peaks.index(g))
                            break_flag = 1
                            break
                    if break_flag == 1:
                        break
        print("The amount of peaks is {} out of {} genotypes". format(len(peaks), len(genotypes)))
    
        print("calculating the valleys")
        
        valleys = list(genotypes)

        for r, g in enumerate(genotypes):
            
            if r%2000==0:
                print("so far went over {} genotypes". format(r))                    
        
            
            break_flag = 0
            if fitness[g]<=lower_bound:
                valleys.pop(valleys.index(g))
            else:
                for i in range(len(g)):
                    for j in _possible_mut_per_position(g, i):
                        if fitness[j] and fitness[j] < fitness[g] + delta:
                            valleys.pop(valleys.index(g))
                            break_flag = 1
                            break
                    if break_flag == 1:
                        break
    
        print("The amount of valleys is {} out of {} genotypes". format(len(valleys), len(genotypes)))    
        
        return peaks, valleys
    
    def hamming(self, g1, g2):
        
        assert len(g1) == len(g2)
        return sum(g1i != g2i for g1i, g2i in zip(g1, g2))
    
    def peaks_vallyes_plot(self, peaks, valleys, x_lim, y_lim, center = 0):
        
        '''
        Goal - plot peaks and valleys around the givven center
        
        Input:
        
        center - integer, the center index inside self.data around which we will show the extrema pts. default - 0. 
        peaks, valles - lists that were computed by the relevant function
        x_lim, y_lim - tuples, the x and y intervals in which the pts will be plotted
        
        Output:
        
        plotting in 2D the peaks and valleys around the center given. 
        and a handle to the fig and axes. 
        '''
        fitness = collections.defaultdict(float) # Here we create a dict that will return a float 0.0 if the key you requested doesnt exist
        for i, row in self.data.iterrows():
            fitness[row['Genotype']] = row['Fit']

        wild_type = self.data.loc[center, 'Genotype']
        genotypes = tuple(fitness.keys())
        assert wild_type in genotypes
        
        
        def _hamming(g1, g2=wild_type):
            assert len(g1) == len(g2)
            return sum(g1i != g2i for g1i, g2i in zip(g1, g2))
        
        peak_distances = [_hamming(p) for p in peaks]
        valley_distances = [_hamming(p) for p in valleys]
    
        peaks_values = [fitness[p] for p in peaks]
        valleys_values = [fitness[p] for p in valleys]

        fig, ax = plt.subplots()
        ax.plot(peak_distances, tuple(peaks_values), '^')
    #    for x,y,z in zip(peak_distances, peaks_values, peaks):
    #        ax.text(x, y, z, fontsize=8)

        ax.plot(valley_distances, valleys_values, 'v')
    #    for x,y,z in zip(valley_distances, valleys_values, valleys):
    #        if y > 0.5:
    #            ax.text(x, y, z, fontsize=8)

        ax.set(xlim=x_lim, 
               ylim=y_lim,
               xlabel="Hamming distance from Center",
               ylabel="Fitness"
        )
        return fig, ax

    # Write a better peaks function
    def centered_pie_plot(self, center = 0, hamming_radius = 4, fitness_list = ['Fit'], three_thresholds = [0.6, 0.9, 1.05]):
        [t1,t2,t3] = three_thresholds
        f = h5py.File(self.stacked_diff_matrix_filename, mode='r')
        plt.figure(figsize=(8,8))
        for i, fit in enumerate(fitness_list):
            plt.subplot(len(fitness_list),1,i+1)
            for r in range(hamming_radius,0,-1):
                neig_index = np.where(f[self.access_root][center, :]==r)[0]
                print('the amount of N{} in {} is {}'.format(r, fit, len(list(neig_index))))
                fitness_vals = self.data.loc[neig_index][fit]
                very_bad = len(list(np.where(fitness_vals <=t1)[0]))
                bad = len(list(np.where((fitness_vals<=t2)&(fitness_vals>t1))[0]))
                neutral = len(list(np.where((fitness_vals<=t3)&(fitness_vals>t2))[0]))
                very_good = len(list(np.where((fitness_vals>=t3))[0]))
                counts = [very_bad, bad, neutral, very_good]
                plt.pie(counts, colors = ['firebrick','plum', 'skyblue', 'lightgreen'], radius=r*0.2)
            plt.title('N1,2,3,4 of {}'.format(fit))
            tight_layout()
        plt.show()
        f.close()
        return None

    def ranks_calculator(self, fitness_list = ['Fit']):        
        for fit in fitness_list:
            self.data['ranked '+fit] = ss.rankdata(self.data[fit])
        avg = self.data['ranked '+fitness_list[0]]
        if len(fitness_list)>0:
            for fit in fitness_list[1:]:
                avg = avg + self.data['ranked ' + fit]
        self.data['avg_rank'] = avg/len(fitness_list)

    def NFD_best_fit_calculator(self, neig_threshold = 10, radius = 1, fit_column = 'Fit', distribution = 'nomral', p = False, plotting_frequency = 150):
        '''
        Assumptions - 
        
        1. You already calculated the entire hamming distance matrix 
        and its filenames is stored at self.stacked_diff_matrix_filename. 
        
        2. You already calculated stability and radius-neighbors
        
        Input - 
        
        
        
        '''
        
        f = h5py.File(self.stacked_diff_matrix_filename, mode='r')
        high_neig_index = self.data[self.data[str(radius) + '-neig']>neig_threshold].index # taking all those that are above threshold
           
        print('There are {} variants that have more than {} neigs. Starting to calculate their NFD'.format(len(list(high_neig_index)), neig_threshold))
        
        lambda_list = []
        mse_list = []

        for j, i in enumerate(high_neig_index):
            neig_index = np.where(f[self.access_root][i, :]==radius)[0] # looking at all the neigs, per index above
            fitness_vals = self.data.loc[neig_index][fit_column] # looking at the neigs fitness
            fitness_range = [np.min(fitness_vals), np.max(fitness_vals)]
            rX = np.linspace(*fitness_range,150) 
            # trying to fit it to some distribution
            if distribution == 'exp':
                loc, scale = ss.expon.fit(fitness_vals)
                lamb = 1/ss.expon.mean(loc = loc, scale = scale)
                lambda_list.append(lamb)                
                rP = ss.expon.pdf(rX, loc = loc, scale = scale)

            elif distribution == 'normal':
                loc, scale = ss.norm.fit(fitness_vals)
                rP = ss.norm.pdf(rX, loc = loc, scale = scale)
   
            fig = plt.figure()
            n, bins, patches = plt.hist(fitness_vals, bins = rX, normed = True)
            
            if plot:        
                if j%plotting_frequency==0:
                    plt.xlabel('fitness')
                    plt.ylabel('amount')
                    plt.title('normed fitness distribution for index {}, #1-neig is {}'.format(i, len(neig_index)))
                    plt.show()
                else:
                    plt.close(fig)
            else:
                plt.close(fig)
            n = np.append(n, 0)
            mse = trapz((rP-n)**2, rX)
            mse_list.append(mse)
            
        f.close()
        
        return mse_list

    def NFD_plotter(self, center = 0, radius = 1):
    
        '''
        Assumptions - 
        
        1. You already calculated the entire hamming distance matrix 
        and its filenames is stored at self.stacked_diff_matrix_filename. 
        
        2. You already calculated stability and radius-neighbors
                
        '''
        
        f = h5py.File(self.stacked_diff_matrix_filename, mode='r')

        neig_index = np.where(f[self.access_root][center, :]==radius)[0] # looking at all the neigs, per index above
        fitness_vals = self.data.loc[neig_index][fit_column] # looking at the neigs fitness
        fitness_range = [np.min(fitness_vals), np.max(fitness_vals)]
        rX = np.linspace(*fitness_range,150) 
        fig = plt.figure()
        n, bins, patches = plt.hist(fitness_vals, bins = rX, normed = True)
        plt.xlabel('Fitness')
        plt.ylabel('Frequency')
        plt.title('normed fitness distribution for index {}, #1-neig is {}'.format(canter, len(neig_index)))
        plt.show()
        f.close()
        
        return 1

    def genotype_flatness_plotting_by_fitness(self,  fitness_intervals = [(0.5,0.95, 'r'), (1.05,2, 'g'), (0.95, 1.05, 'b')], special_indexes = [(0,'aqua')],
                                              neig_threshold = 10, column_fit = 'Fit', r = 1):

        '''
        Assumptions  - You already ran stability_calculator for r-neigs
        
        
        '''
        data = self.data
        for u, v, c in fitness_intervals:
            plt.scatter(data[(data[str(r)+'-neig']>neig_threshold) & (data[column_fit]>u) & (data[column_fit]<v)][str(r)+'-abs-grad-mean-'+column_fit], 
                        data[(data[str(r)+'-neig']>neig_threshold) & (data[column_fit]>u) & (data[column_fit]<v)][str(r)+'-grad-std-abs-'+column_fit], 
                        c = c, label = str(v)+'>'+column_fit+'>'+str(u), marker = '+')
        
        for u, c in special_indexes:
            plt.scatter(data.loc[u,str(r)+'-abs-grad-mean-'+column_fit], data.loc[u,str(r)+'-grad-std-abs-'+column_fit], c = c, 
                        label = 'genotype {}'.format(u), marker = 'o', s = 100)

        plt.title('Variance against Expectation of the Absolute-Value Gradients Vector \n Only variants w/ more than {} neig presented, Env={}'.format(neig_threshold, column_fit))
        plt.xlabel('Expectation')
        plt.ylabel('Variance')
        plt.legend()
        return 1

    def genotype_flatness_plotting_by_neigs(self, neigs = [(0, 'aqua'),(1, 'r'), (2,'g'), (3,'k'), (4,'b')], neig_threshold = 10, column_fit = 'Fit', r = 1):
        
        '''
        depycting flatness along rnaging hamming distances from the WT
        '''
        
        data = self.data
        for u, c in neigs:
            plt.scatter(data[(data[str(r)+'-neig']>neig_threshold) & (data['Num']==u)][str(r)+'-abs-grad-mean-'+column_fit], 
                        data[(data[str(r)+'-neig']>neig_threshold) & (data['Num']==u)][str(r)+'-grad-std-abs-'+column_fit], c = c, label = '$N_{}$'.format(u), marker = '+')
   
        plt.title('Variance against Expectation of the Absolute-Value Gradients Vector \n Only variants w/ more than {} neig presented, Env={}'.format(t, column_fit))
        plt.xlabel('Expectation')
        plt.ylabel('Variance')
        plt.legend()
        
        return 1

        
        
        from scipy.stats.stats import pearsonr


        
'''        
t = 50
plt.subplot(2,1,1)
r, p = pearsonr(jj[(jj['1-neig']>t)]['1-abs-grad-mean-'+column_fit], jj[(jj['1-neig']>t)]['1-neig'])
plt.scatter(jj[(jj['1-neig']>t)]['1-abs-grad-mean-'+column_fit], jj[(jj['1-neig']>t)]['1-neig'])
plt.title('Avg Vs 1-Neig. Threshold = {} \n r = {}, p = {}'. format(t,r,p))
plt.xlabel('Avg. of gradients')
plt.ylabel('#1-neig')

plt.subplot(2,1,2)
r, p = pearsonr(jj[(jj['1-neig']>t)]['1-grad-std-abs-'+column_fit], jj[(jj['1-neig']>t)]['1-neig'])
plt.scatter(jj[(jj['1-neig']>t)]['1-grad-std-abs-'+column_fit], jj[(jj['1-neig']>t)]['1-neig'])
plt.title('Var Vs 1-Neig. Threshold = {} \n r = {}, p = {}'. format(t,r,p))
plt.xlabel('Var of gradients')
plt.ylabel('#1-neig')

tight_layout()

trna_e_pipe.data['Fit23-normalized'] = trna_e_pipe.data['Fit23'].apply(lambda x: x*0.5)
trna_e_pipe.data['Fit37-normalized'] = trna_e_pipe.data['Fit37']/1.1111
trna_e_pipe.data['FitDMSO-normalized'] = trna_e_pipe.data['FitDMSO']/1.064
trna_e_pipe.data['FitDMSO-gradient'] = np.abs(trna_e_pipe.data['Fit30'] - trna_e_pipe.data['FitDMSO-normalized'])
trna_e_pipe.data['Fit37-gradient'] = np.abs(trna_e_pipe.data['Fit30'] - trna_e_pipe.data['Fit37-normalized'])
trna_e_pipe.data['Fit23-gradient'] = np.abs(trna_e_pipe.data['Fit30'] - trna_e_pipe.data['Fit23-normalized'])

trna_e_pipe.data['Env_mean'] = trna_e_pipe.data[['FitDMSO-gradient', 'Fit37-gradient', 'Fit23-gradient']].mean(axis=1)
trna_e_pipe.data['Env_median'] = trna_e_pipe.data[['FitDMSO-gradient', 'Fit37-gradient', 'Fit23-gradient']].std(axis=1)
trna_e_pipe.data['Env_std'] = trna_e_pipe.data[['FitDMSO-gradient', 'Fit37-gradient', 'Fit23-gradient']].median(axis=1)

# trna_e_pipe.data['Env_mean'] = trna_e_pipe.data[['FitDMSO-gradient', 'Fit37-gradient']].mean(axis=1)
# trna_e_pipe.data['Env_median'] = trna_e_pipe.data[['FitDMSO-gradient', 'Fit37-gradient']].std(axis=1)
# trna_e_pipe.data['Env_std'] = trna_e_pipe.data[['FitDMSO-gradient', 'Fit37-gradient']].median(axis=1)


jj = trna_e_pipe.data
t = 0.5
t1 = 0.95
t2 = 1.05
n = 9
plt.scatter(jj[(jj['Fit30']>t) & (jj['Num']<n)]['Env_mean'] , jj[(jj['Fit30']>t) & (jj['Num']<n)]['Env_std'], c = 'r', label = 'fittnes>{}'.format(t), marker = '+')
# plt.scatter(jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['Env_mean'] , jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['Env_std'], c = 'r', label = 'fittnes>{}'.format(t), marker = '+')
plt.scatter(jj.loc[0, 'Env_mean'] , jj.loc[0, 'Env_std'], c = 'aqua', label = 'WT', marker = 'o', s = 100)

# plt.scatter(jj[(jj['1-neig']>t) & (jj[column_fit]<0.9)]['1-abs-grad-mean-'+column_fit], jj[(jj['1-neig']>t) & (jj[column_fit]<0.9)]['1-grad-std-abs-'+column_fit], c = 'k', label = 'fitt<0.9', marker = '+')
# plt.scatter(jj[(jj['1-neig']>t) & (jj[column_fit]<1) & (jj[column_fit]>0.9)]['1-abs-grad-mean-'+column_fit], jj[(jj['1-neig']>t) & (jj[column_fit]<1) & (jj[column_fit]>0.9)]['1-grad-std-abs-'+column_fit], c = 'g', label = '0.9<fitt<1', marker = '+')
# plt.scatter(jj[(jj['1-neig']>t) & (jj[column_fit]==1)]['1-abs-grad-mean-'+column_fit], jj[(jj['1-neig']>t) & (jj[column_fit]==1)]['1-grad-std-abs-'+column_fit], c = 'aqua', label = 'WT', marker = 'o', s = 100)
# plt.scatter(jj.loc[194,'1-abs-grad-mean-'+column_fit], jj.loc[194,'1-grad-std-abs-'+column_fit], c = 'brown', label = '1-mutant-fit=~1', marker = 'o', s = 100)
# plt.scatter(jj.loc[85,'1-abs-grad-mean-'+column_fit], jj.loc[85,'1-grad-std-abs-'+column_fit], c = 'pink', label = '1-mutant-fit=~0.8', marker = 'o', s = 100)
# plt.scatter(jj.loc[4658,'1-abs-grad-mean-'+column_fit], jj.loc[4658,'1-grad-std-abs-'+column_fit], c = 'lime', label = '2-mutant-fit=~1', marker = 'o', s = 100)
plt.title('Variance against Expectation of the Absolute-Value Gradients Vector across environments'.format(t, column_fit))
plt.xlabel('Expectation')
plt.ylabel('Variance')
plt.legend()

plt.figure(figsize=(8,8))

plt.subplot(2,1,1)
jj[jj['Fit30']>=1]['Env_mean'].hist(bins = np.linspace(0,0.5,150), color = 'r', histtype='step', label = '$Fit30\geq1$', normed=True)
jj[(jj['Fit30']>1.05) & (jj['Fit30']<1.2)]['Env_mean'].hist(bins = np.linspace(0,0.5,150), color = 'k', histtype='step', label = '$1.2>Fit30\geq1.05$',  normed=True)
jj[(jj['Fit30']>=1.2)]['Env_mean'].hist(bins = np.linspace(0,0.5,150), color = 'g', histtype='step', label = '$Fit30\geq1.2$',  normed=True)
jj[(jj['Fit30']>=0.5) & (jj['Fit30']<1)]['Env_mean'].hist(bins = np.linspace(0,0.5,150), color = 'b', histtype='step', label = '$1>Fit30\geq0.5$',  normed=True)
plt.vlines(jj.loc[0,'Env_mean'], ymin=0, ymax=17.5, color = 'aqua', label = 'WT Env_mean')
plt.legend(fontsize = 'large')
plt.title('Env. mean flatness')
plt.xlabel('mean of  |Env. gradient|')
plt.ylabel('frequency')

plt.subplot(2,1,2)
jj[jj['Fit30']>=1]['Env_std'].hist(bins = np.linspace(0,0.5,150), color = 'r', histtype='step', label = '$Fit30\geq1$', normed=True)
jj[(jj['Fit30']>1.05) & (jj['Fit30']<1.2)]['Env_std'].hist(bins = np.linspace(0,0.5,150), color = 'k', histtype='step', label = '$1.2>Fit30\geq1.05$',  normed=True)
jj[(jj['Fit30']>=1.2)]['Env_std'].hist(bins = np.linspace(0,0.5,150), color = 'g', histtype='step', label = '$Fit30\geq1.2$',  normed=True)
jj[(jj['Fit30']>=0.5) & (jj['Fit30']<1)]['Env_std'].hist(bins = np.linspace(0,0.5,150), color = 'b', histtype='step', label = '$1>Fit30\geq0.5$',  normed=True)
plt.vlines(jj.loc[0,'Env_std'], ymin=0, ymax=17.5, color = 'aqua', label = 'WT Env_std')
plt.legend(fontsize = 'large')
plt.title('Env. flatness std')
plt.xlabel('Variance')
plt.ylabel('frequency')

tight_layout()

plt.figure(figsize=(8,8))

plt.subplot(2,1,1)
jj[jj['Num']==1]['Env_mean'].hist(bins = np.linspace(0,0.5,150), color = 'r', histtype='step', label = '$N_1$', normed=True)
jj[jj['Num']==2]['Env_mean'].hist(bins = np.linspace(0,0.5,150), color = 'k', histtype='step', label = '$N_2$',  normed=True)
jj[jj['Num']==3]['Env_mean'].hist(bins = np.linspace(0,0.5,150), color = 'g', histtype='step', label = '$N_3$',  normed=True)
jj[jj['Num']==4]['Env_mean'].hist(bins = np.linspace(0,0.5,150), color = 'b', histtype='step', label = '$N_4$',  normed=True)
plt.vlines(jj.loc[0,'Env_mean'], ymin=0, ymax=17.5, color = 'aqua', label = 'WT Env_mean')
plt.legend(fontsize = 'large')
plt.title('Env. mean flatness')
plt.xlabel('mean of  |Env. gradient|')
plt.ylabel('frequency')


plt.subplot(2,1,2)
jj[jj['Num']==1]['Env_std'].hist(bins = np.linspace(0,0.5,150), color = 'r', histtype='step', label = '$N_1$', normed=True)
jj[jj['Num']==2]['Env_std'].hist(bins = np.linspace(0,0.5,150), color = 'k', histtype='step', label = '$N_2$',  normed=True)
jj[jj['Num']==3]['Env_std'].hist(bins = np.linspace(0,0.5,150), color = 'g', histtype='step', label = '$N_3$',  normed=True)
jj[jj['Num']==4]['Env_std'].hist(bins = np.linspace(0,0.5,150), color = 'b', histtype='step', label = '$N_4$',  normed=True)
plt.vlines(jj.loc[0,'Env_std'], ymin=0, ymax=17.5, color = 'aqua', label = 'WT Env_std')
plt.legend(fontsize = 'large')
plt.title('Env. flatness std')
plt.xlabel('Variance')
plt.ylabel('frequency')

tight_layout()

t1 = 0.95
t2 = 1.05
x = np.asarray(jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['1-abs-grad-mean-Fit30'])
y = np.asarray(jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['Env_mean'])
y = y[np.logical_not(np.isnan(x))]
x = x[np.logical_not(np.isnan(x))]
r, p = ss.pearsonr(x, y)
plt.scatter(jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['1-abs-grad-mean-Fit30'], jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['Env_mean'], c= jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['Fit30'], cmap=matplotlib.cm.rainbow, label = '${}>Fit30>{}$'.format(t2,t1), marker = '+')
# plt.scatter(jj.loc[0, '1-abs-grad-mean-Fit30'], jj.loc[0, 'Env_mean'], c = 'aqua', label = 'WT', marker = 'o', s = 100)
plt.title('Environmental Flatness VS Genotype Flatness, ${}>Fit30>{}$, \n r={}, p = {}'.format(t2, t1, r ,p))
plt.xlabel('Genotype Flatness')
plt.ylabel('Environmental Flatness')
cbar = matplotlib.pyplot.colorbar()
cbar.set_label('          Fitness', rotation=0)
y_lims = plt.ylim()
x_lims = plt.xlim()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
x = jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['1-abs-grad-mean-Fit30']
y = jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['Env_mean']
z = jj[(jj['Fit30']>t1) & (jj['Fit30']<t2)]['Fit30']
ax.scatter(x, y, z)
ax.set_xlabel('Gradient Flatness')
ax.set_ylabel('Environmental Flatness')
ax.set_zlabel('Fitness')
ax.view_init(0, 45)
# data = np.arange(10, 0, -1).reshape(10, 1)
# im = ax.imshow(data, cmap='rainbow')
# fig.colorbar(im, orientation='vertical')
plt.show()


iterations = 15
multi_env_results_df = pd.DataFrame(index = range(iterations), columns=['fit_init', 'init_method', 'rand_regime', 'fitness_regime', 'winner',  'fit_end', 'generations'])
for i in range(iterations):
    g_4 = trna_quasispecies_model_tester(15200, trna_extended.data, organism_amount=10**8)
    g_4.Q = R
    fitness_list = ['Fit23-normalized', 'Fit37-normalized', 'FitDMSO-normalized', 'Fit30-normalized']
    rand_fit_init = np.random.choice(fitness_list)
    init_method =  np.random.choice(['worst', 'nearly_worst', 'nearly_wt'])
    g_4.change_pop_init(pop_init_method=init_method, fit_column = rand_fit_init)
    rand_regime = np.random.randint(3,8,4)
    np.random.shuffle(fitness_list)
    g_4.multi_env_iterator(freq_list = rand_regime, fls_list = fitness_list, w_pop=True);
    winner_index = np.argmax(g_4.p_list[-1])
    
    multi_env_results_df.loc[i,'fit_init'] = rand_fit_init
    multi_env_results_df.loc[i,'fit_end'] = g_4.current_fls
    multi_env_results_df.loc[i,'generations'] = len(g_4.p_list)
    multi_env_results_df.loc[i,'init_method'] = init_method
    multi_env_results_df.loc[i,'rand_regime'] = rand_regime
    multi_env_results_df.loc[i,'fitness_regime'] = fitness_list
    multi_env_results_df.loc[i,'winner'] = winner_index
    
iterations = 200
rand_QS_multienv_results_df = pd.DataFrame(index = range(iterations), columns=['fit_init', 'init_method', 'rand_regime', 'distribution', 'winner', 'winner_fitness', 'generations'])
for i in range(iterations):
    g_4 = trna_quasispecies_model_tester(15200, trna_extended.data, organism_amount=10**8)
    g_4.Q = R
    fitness_list = ['Fit23', 'Fit37', 'FitDMSO', 'Fit30']
    rand_fit_init = np.random.choice(fitness_list)
    init_method =  np.random.choice(['worst', 'nearly_worst', 'nearly_wt'])
    g_4.change_pop_init(pop_init_method=init_method, fit_column = rand_fit_init)
    d = mp.random.choice(['normal', 'exp', 'uniform'])
    rand_regime = np.random.randint(3,8,4)
    g_4.multi_env_iterator(freq_list = rand_regime, fls_list = fitness_list, w_pop=True, random_method = 'totaly_random_interval',  distribution = d);
    winner_index = np.argmax(g_4.p_list[-1])
    winner_fitness = g_4.w[winner_index]
    
    rand_QS_multienv_results_df.loc[i,'fit_init'] = rand_fit_init
    rand_QS_multienv_results_df.loc[i,'init_method'] = init_method
    # The above two don't need to matter - if they are - it means that the amount of trajectories is interfering with the results.
    rand_QS_multienv_results_df.loc[i,'rand_regime'] = rand_regime
    rand_QS_multienv_results_df.loc[i,'generations'] = len(g_4.p_list)
    rand_QS_multienv_results_df.loc[i,'distribution'] = d
    rand_QS_multienv_results_df.loc[i,'winner'] = winner_index
    rand_QS_multienv_results_df.loc[i,'winner_fitness'] = winner_fitness
    
'''