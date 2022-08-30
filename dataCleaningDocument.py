'''
Document for cleaning the data for input into the machine learning algorithm

By: Chasing Midnight - Antony Georgiadis

'''


'''
Goals:
Go through each column and reformat NaNs and -'s into new values that will allow for further formating

Steps:
1) Pull Data from file
2) Figure out how to step through columns and read column data
3) Pull the column data and store it again
4) Get a mapping of the things present and frequency as a paired list
5) Read and replace the NaNs and -'s
'''

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import all files for parsing

# BC_train = pd.read_csv('test_data/tfbs_score_BC217_train.tsv', sep = '\t')
# YJ_train = pd.read_csv('test_data/tfbs_score_YJF153_train.tsv', sep = '\t' )
def understandData():
    train_features = pd.read_csv('train_data/features_train.tsv', sep = '\t')
    train_labels = train_features.pop('DIFF')
    def typeAnalysis(typesArr, obj):
        i = 0
        for typeComp in typesArr:
            if isinstance(obj, typeComp):
                return False, i
            i += 1
        return True, i

    singleTypeColumns = []
    multiTypeCols = []

    train_features["pos_exon"].str.split("-",expand = True)
    # Go through columns check the type distributions
    for column in train_features.columns :
        # Unique values
        unique_values = train_features[column].unique()
        length = len(unique_values)

        numTypes = 0
        types = []
        typeCount = []

        for value in unique_values:
            newType, typeLoc = typeAnalysis(types, value)
            if newType:
                types.append(type(value))
                typeCount.append(0)
                numTypes += 1
            typeCount[typeLoc] += 1
            
        print("Number of Types is:" + str(numTypes) + ", and types are:" + str(types))
        print("Type Count:" + str(typeCount))
        if numTypes > 1:
            multiTypeCols.append(column)
        else:
            singleTypeColumns.append(column)

        if length < 20: 
            pass
            print(column + ' : ', length , '\n')

        else:
            pass
            # print(unique_values[0:20])
            print(column + ' : ', length , '\n')
    print(singleTypeColumns)
    print(multiTypeCols)


# Convert the Aminochains and locations into otherDatapoints


if __name__ == "__main__":
    # Split the column with dashed values and add them to the end of the data set
    train_features = pd.read_csv('train_data/features_train.tsv', sep = '\t')
    train_labels = train_features.pop('DIFF')
    # Split based on non-numeric values
    train_features[["pos_exon1_start", "pos_exon1_end", "pos_exon2_start", "pos_exon2_end", "pos_exon3_start", "pos_exon3_end"]] = train_features["pos_exon"].str.split(r'\W', expand = True)
    print(train_features["pos_exon"].str.split(r'\W', expand = True).shape)
    train_features.drop(columns='pos_exon')

    # Convert all columns that are possible into floats and print to screen a list where not possible
    failFloat32 = []
    for column in train_features.columns:
        try: 
            train_features[column].astype(np.float32)
        except:
            failFloat32.append(column)
    
    # Replace all Nan and Na values with 0
    train_features.fillna(0)

    

    





# Figure out some info about the genotype data 
genotype_data = ['YJF153_geno_SNP_promoter', 'BC217_geno_SNP_promoter',
                 'YJF153_geno_MIXED_promoter','BC217_geno_MIXED_promoter',
                 'YJF153_geno_SNP_3end','BC217_geno_SNP_3end','YJF153_geno_MIXED_3end',
                 'BC217_geno_MIXED_3end', 'YJF153_geno_SNPs_gene','BC217_geno_SNPs_gene',
                 'YJF153_geno_MIXED_gene','BC217_geno_MIXED_gene']
def checkData(train_features, genotype_data):
    largest = 0
    running_total = 0 
    running_num = 0 

    chunkNums = []
    maxChunkSize = 0
    chunkSizes = []
    # Check for number of splits in each column
    for column in genotype_data:
        for i in range(train_features.shape[0]):
            try:
                current = len(train_features[column][i].split(':'))
                running_total = running_total + current 
                running_num =  running_num + 1
                for val in train_features[column][i].split(':'):
                    size = len(val)
                    if size > maxChunkSize: maxChunkSize=size
                    chunkSizes.append(size)
                if current != 0:
                    chunkNums.append(current)
            except: continue
            if (largest == 0) | (current > largest) :
                largest = current
        
    print('largest:' , largest)
    print('total:', running_total, 'num:',running_num)
    print('avg:' ,running_total/running_num)
    print('maxChunkSize:', maxChunkSize)
    plt.hist(chunkNums, bins=278)
    plt.show()
    plt.hist(chunkSizes, bins=142)
    plt.show()

# Check for unique chunk
def checkIfNewChunk(existingChunks, chunk):
    if chunk in existingChunks:
        return False
    else:
        return True

# Check for the number of unique chunks of strings
splitCap = 20
def extractSequenceChunks(train_features, genotype_data):
    unique_seq_chunks = ['A','T','C','G']
    for column in genotype_data:
        for i in range(train_features.shape[0]):
            try:
                getChunks = train_features[column][i].split(":")
                for chunk in getChunks:
                    newChunk = checkIfNewChunk(unique_seq_chunks, chunk)
                    if newChunk:
                        unique_seq_chunks.append(chunk)
            except: continue
    np.save('uniqueSeqChunks', unique_seq_chunks)
    print('numUniqueSeq:', len(unique_seq_chunks))


def encodeGenotypes(value, uniqueSeqChunks):
    try:
        return uniqueSeqChunks.index(value)
    except:
        return len(uniqueSeqChunks)

# Take input of the feature set and the genotype_data as well as the cutoff size
def translateColumnChunks(train_features, genotype_data, uniqueSeqChunks ,cutoffSize=20):
    rows = train_features.shape[0]
    # Go through each column
    for column in genotype_data:
        # Allocate space for each translation of 20
        translatedColumn = np.zeros((rows, cutoffSize))
        for i in range(rows):
            if type(train_features[column][i]) != str:
                getChunks = [0]
            else:
                getChunks = train_features[column][i].split(":")
            for j in range(cutoffSize):
                try:
                    translatedColumn[i][j] = encodeGenotypes(getChunks[j], uniqueSeqChunks)+1
                except:
                    translatedColumn[i][j] = 0
        train_features.drop(columns=column, inplace=True)
        for i in range(cutoffSize):
            columnAdding = column + "_SplitPart_" + str(i)
            train_features[columnAdding] = translatedColumn[:,i].tolist()
    return train_features
    

        
numerical_lists = ['dist_SNP_promoter', 'dist_MIXED_promoter','dist_SNPs_gene', 'dist_MIXED_gene']

def translateDistChunks(train_features, numerical_lists, cutoffSize=20):
    rows = train_features.shape[0]
    # Go through each column
    for column in numerical_lists:
        # Allocate space for each translation of 20
        translatedColumn = np.zeros((rows, cutoffSize), dtype=np.float32)
        for i in range(rows):
            if type(train_features[column][i]) != str:
                getChunks = [0]
            else:
                getChunks = train_features[column][i].split(":")
            for j in range(cutoffSize):
                try:
                    translatedColumn[i][j] = np.float32(getChunks[j])
                except:
                    translatedColumn[i][j] = 0
        train_features.drop(columns=column, inplace=True)
        for i in range(cutoffSize):
            columnAdding = column + "_SplitPart_" + str(i)
            train_features[columnAdding] = translatedColumn[:,i].tolist()
    return train_features

uniqueSeqChunks = np.load("uniqueSeqChunks.npy")
train_features_translated = translateColumnChunks(train_features, genotype_data, uniqueSeqChunks)
train_features_translated = translateDistChunks(train_features_translated, numerical_lists)