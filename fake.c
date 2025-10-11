#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <ctype.h> // For tolower

// =================================================================
//                            FND_DEFS.H
// =================================================================

// --- Constants ---
#define MAX_WORD_LEN 50
#define VOCAB_SIZE 20000 // Hash Table size
#define MAX_FEATURES 10000
#define NUM_CLASSES 2 // Fake and Real
#define REAL_CLASS 0
#define FAKE_CLASS 1

// --- 1. Core Data Structures ---

// Simple Linked List Node (for Hash Table collision resolution)
typedef struct HashNode {
    char word[MAX_WORD_LEN];
    int word_id; // Unique ID for each word
    int count;   // Global frequency
    struct HashNode *next;
} HashNode;

// Hash Table / Vocabulary (Array of Linked Lists)
typedef struct {
    HashNode *table[VOCAB_SIZE];
    int total_words; // Total unique words in the vocabulary
} HashTable;

// Document structure (Simplified version for Array/Dynamic Array)
typedef struct {
    int id;
    int label; // 0 for Real, 1 for Fake
    char *text; // Pointer to the article text
    // Feature representation (used as a sparse row/dynamic array)
    int *feature_indices; // Indices into the vocabulary/sparse matrix
    double *feature_values; // TF or TF-IDF values (full vector for simplicity)
    int feature_count;
} Document;

// Dataset (Dynamic Array of Documents)
typedef struct {
    Document *documents;
    int size;
    int capacity;
} Dataset;

// Sparse Matrix Node (Conceptual - defined but not fully used for MNB)
typedef struct SparseNode {
    int doc_id;
    int feature_id;
    double value;
    struct SparseNode *next_in_row;
    struct SparseNode *next_in_col;
} SparseNode;

// --- 2. Naive Bayes Model Structures ---

typedef struct {
    double log_prior[NUM_CLASSES];               // P(C) - Log prior probability for each class
    double log_likelihood[VOCAB_SIZE][NUM_CLASSES]; // P(W|C) - Log likelihood for each word in each class
    long long class_total_word_count[NUM_CLASSES];  // N_c - Total count of all words in class c
    long long global_total_word_count;              // Sum of N_c
    int vocabulary_size;                          // |V| - Size of the vocabulary (for Laplace smoothing)
} MultinomialNaiveBayes;

// --- Function Prototypes (now defined before use) ---
unsigned int hash(const char *str);
HashNode *ht_insert(HashTable *ht, const char *word);
void tokenize(const char *text, HashTable *vocab, Document *doc);
void build_vocabulary(Dataset *data, HashTable *vocab);
Dataset *read_csv_data(const char *filepath);
void calculate_tf(Document *doc, HashTable *vocab);
void calculate_tfidf(Dataset *data, HashTable *vocab); // Conceptual
void train_naive_bayes(Dataset *train_data, HashTable *vocab, MultinomialNaiveBayes *model);
int predict_naive_bayes(const char *text, HashTable *vocab, MultinomialNaiveBayes *model);
void cleanup_dataset(Dataset *data);
void cleanup_vocab(HashTable *ht);
void cleanup_model(MultinomialNaiveBayes *model);


// =================================================================
//                            FND_UTILS.C
// =================================================================

// --- Hash Function (Djb2 simple string hash) ---
unsigned int hash(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % VOCAB_SIZE;
}

// --- Hash Table Insertion ---
HashNode *ht_insert(HashTable *ht, const char *word) {
    unsigned int index = hash(word);
    HashNode *current = ht->table[index];

    // Check if word already exists
    while (current != NULL) {
        if (strcmp(current->word, word) == 0) {
            current->count++;
            return current;
        }
        current = current->next;
    }

    // Word does not exist, insert new node
    if (ht->total_words >= VOCAB_SIZE) {
        // Prevent buffer overflow if vocab is huge, although VOCAB_SIZE is large.
        // In a real sparse matrix setup, this wouldn't be necessary.
        return NULL;
    }
    
    HashNode *new_node = (HashNode *)malloc(sizeof(HashNode));
    if (new_node == NULL) {
        perror("Failed to allocate memory for HashNode");
        exit(EXIT_FAILURE);
    }
    strncpy(new_node->word, word, MAX_WORD_LEN - 1);
    new_node->word[MAX_WORD_LEN - 1] = '\0';
    new_node->count = 1;
    new_node->word_id = ht->total_words++; // Assign unique ID
    new_node->next = ht->table[index];
    ht->table[index] = new_node;

    return new_node;
}

// --- Conceptual CSV Reader (Highly Simplified) ---
Dataset *read_csv_data(const char *filepath) {
    printf("--- NOTE: Conceptual CSV Reader (File: %s) ---\n", filepath);
    printf("Using a small hardcoded dataset for demonstration of the ML flow.\n");
    
    // For demonstration, we will manually create a tiny dataset:
    Dataset *data = (Dataset *)malloc(sizeof(Dataset));
    data->size = 4;
    data->capacity = 4;
    data->documents = (Document *)malloc(data->capacity * sizeof(Document));
    
    // Document 1 (Real)
    data->documents[0].id = 0;
    data->documents[0].label = REAL_CLASS;
    data->documents[0].text = strdup("The president visited New York and held a meeting.");
    data->documents[0].feature_indices = NULL; data->documents[0].feature_values = NULL; data->documents[0].feature_count = 0;

    // Document 2 (Fake)
    data->documents[1].id = 1;
    data->documents[1].label = FAKE_CLASS;
    data->documents[1].text = strdup("Amazing fact: New York is the best city to live.");
    data->documents[1].feature_indices = NULL; data->documents[1].feature_values = NULL; data->documents[1].feature_count = 0;
    
    // Document 3 (Real)
    data->documents[2].id = 2;
    data->documents[2].label = REAL_CLASS;
    data->documents[2].text = strdup("The president gave a speech about the economy.");
    data->documents[2].feature_indices = NULL; data->documents[2].feature_values = NULL; data->documents[2].feature_count = 0;
    
    // Document 4 (Fake)
    data->documents[3].id = 3;
    data->documents[3].label = FAKE_CLASS;
    data->documents[3].text = strdup("This city is totally fake and the best city.");
    data->documents[3].feature_indices = NULL; data->documents[3].feature_values = NULL; data->documents[3].feature_count = 0;

    printf("Loaded %d sample documents for demonstration.\n", data->size);
    return data;
}

// --- Tokenization and Vocabulary Building (Simplified) ---
void tokenize(const char *text, HashTable *vocab, Document *doc) {
    char *text_copy = strdup(text);
    char *saveptr = text_copy;
    char *token;
    
    // Define delimiters (spaces and common punctuation)
    char delimiters[] = " .,!?\"'():-;"; 

    int current_capacity = 20;
    int current_count = 0;
    int *temp_indices = (int *)malloc(current_capacity * sizeof(int));
    
    // strtok is not thread-safe, but works here. Use strtok_r if needed.
    token = strtok(text_copy, delimiters); 
    
    while (token != NULL) {
        // Simple case conversion
        for (int i = 0; token[i]; i++) {
            token[i] = tolower((unsigned char) token[i]);
        }
        
        // Skip tokens that are too short (simple preprocessing)
        if (strlen(token) > 2) {
            HashNode *node = ht_insert(vocab, token);

            // Add word ID to the document's token stream
            if (node != NULL) {
                 if (current_count == current_capacity) {
                    current_capacity *= 2;
                    temp_indices = (int *)realloc(temp_indices, current_capacity * sizeof(int));
                    if (temp_indices == NULL) {
                        perror("Realloc failed in tokenize");
                        exit(EXIT_FAILURE);
                    }
                }
                temp_indices[current_count++] = node->word_id;
            }
        }
        
        token = strtok(NULL, delimiters);
    }
    
    free(saveptr); // Free the strdup copy
    
    // Finalize document features (indices/token stream)
    doc->feature_indices = temp_indices;
    doc->feature_count = current_count;
    // Shrink the array to the actual size
    doc->feature_indices = (int *)realloc(doc->feature_indices, current_count * sizeof(int));
}

// --- Build Vocabulary from Dataset ---
void build_vocabulary(Dataset *data, HashTable *vocab) {
    printf("\n--- Building Vocabulary and Tokenizing Documents ---\n");
    for (int i = 0; i < data->size; i++) {
        tokenize(data->documents[i].text, vocab, &data->documents[i]);
    }
    printf("Vocabulary built with %d unique words.\n", vocab->total_words);
}

// --- Calculate Term Frequency (TF) ---
// This function calculates the raw word counts (Bag-of-Words) and stores them 
// in the document's feature_values array for MNB training.
void calculate_tf(Document *doc, HashTable *vocab) {
    if (doc->feature_count == 0) return;

    // Use a calloc array of size vocab->total_words to store TF counts (sparse row)
    doc->feature_values = (double *)calloc(vocab->total_words, sizeof(double));
    if (doc->feature_values == NULL) {
        perror("Calloc failed for TF array");
        exit(EXIT_FAILURE);
    }

    // 1. Count occurrences (Bag-of-Words)
    for (int i = 0; i < doc->feature_count; i++) {
        int word_id = doc->feature_indices[i];
        if (word_id < vocab->total_words) {
            // Store raw count for Multinomial Naive Bayes
            doc->feature_values[word_id]++; 
        }
    }
    
    // The TF vector length is the total unique words in the vocab.
    doc->feature_count = vocab->total_words; 
}

// --- Conceptual TF-IDF Calculation ---
void calculate_tfidf(Dataset *data, HashTable *vocab) {
    // This is conceptual. In a full implementation, you'd calculate:
    // 1. DF (Document Frequency) for all words
    // 2. IDF = log(N / DF)
    // 3. TF-IDF = TF * IDF
    printf("--- NOTE: Conceptual TF-IDF/Sparse Matrix Function Called ---\n");
}


// =================================================================
//                             FND_ML.C
// =================================================================

// --- Train Multinomial Naive Bayes ---
void train_naive_bayes(Dataset *train_data, HashTable *vocab, MultinomialNaiveBayes *model) {
    printf("\n--- Training Multinomial Naive Bayes Model ---\n");
    
    // 1. Initialize counts
    // Use dynamic allocation for class word counts to allow for larger VOCAB_SIZE
    long long (*class_word_counts)[VOCAB_SIZE] = calloc(NUM_CLASSES, sizeof(*class_word_counts));
    if (class_word_counts == NULL) {
        perror("Failed to allocate class_word_counts");
        exit(EXIT_FAILURE);
    }
    
    model->global_total_word_count = 0;
    model->vocabulary_size = vocab->total_words;

    // 2. Count class priors and word occurrences per class
    int class_document_count[NUM_CLASSES] = {0};
    
    for (int i = 0; i < train_data->size; i++) {
        Document *doc = &train_data->documents[i];
        int c = doc->label; // Class label
        
        class_document_count[c]++;
        
        // Calculate TF (Bag-of-Words counts) for this document
        calculate_tf(doc, vocab);
        
        // Sum word counts for this class
        for (int j = 0; j < doc->feature_count; j++) {
            // j is the word_id (index into the feature vector)
            int count = (int)doc->feature_values[j]; // Raw count
            
            if (count > 0) {
                class_word_counts[c][j] += count;
                model->class_total_word_count[c] += count;
            }
        }
    }
    
    model->global_total_word_count = model->class_total_word_count[REAL_CLASS] + model->class_total_word_count[FAKE_CLASS];

    // 3. Calculate Log Priors: log(P(C))
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Laplace smoothing applied to document counts as well
        model->log_prior[c] = log((double)(class_document_count[c] + 1) / (train_data->size + NUM_CLASSES));
    }
    
    // 4. Calculate Log Likelihoods: log(P(W|C)) using Laplace (Additive) Smoothing
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Denominator: Total words in class c + |V| (vocabulary size)
        double denominator = (double)model->class_total_word_count[c] + model->vocabulary_size;
        
        for (int w = 0; w < model->vocabulary_size; w++) {
            // Numerator: Count of word w in class c + 1
            double numerator = (double)class_word_counts[c][w] + 1.0;
            
            model->log_likelihood[w][c] = log(numerator / denominator);
        }
    }

    free(class_word_counts);
    printf("Training complete. Vocabulary size: %d.\n", model->vocabulary_size);
}

// --- Predict with Multinomial Naive Bayes ---
int predict_naive_bayes(const char *text, HashTable *vocab, MultinomialNaiveBayes *model) {
    // 1. Tokenize the input text using the training vocabulary
    Document new_doc = {0};
    // The tokenization will use the existing vocab but will not add new words,
    // as it is only used here to generate word_ids.
    tokenize(text, vocab, &new_doc);

    // 2. Calculate Log Probability for each class
    double log_prob[NUM_CLASSES] = {0.0};
    
    for (int c = 0; c < NUM_CLASSES; c++) {
        // Start with the log prior: log(P(C))
        log_prob[c] = model->log_prior[c];
        
        // Sum the log likelihoods for all words in the document
        // log(P(C|D)) = log(P(C)) + SUM_i(log(P(W_i|C)))
        
        for (int i = 0; i < new_doc.feature_count; i++) {
            int word_id = new_doc.feature_indices[i];
            
            // Only use words present in the training vocabulary
            if (word_id < model->vocabulary_size) {
                log_prob[c] += model->log_likelihood[word_id][c];
            }
        }
    }

    // 3. Cleanup temporary document structures
    if (new_doc.feature_indices) free(new_doc.feature_indices);
    
    // 4. Choose the class with the maximum log probability
    if (log_prob[FAKE_CLASS] > log_prob[REAL_CLASS]) {
        return FAKE_CLASS;
    } else {
        return REAL_CLASS;
    }
}

// --- Cleanup Functions ---
void cleanup_dataset(Dataset *data) {
    for (int i = 0; i < data->size; i++) {
        free(data->documents[i].text);
        if (data->documents[i].feature_indices) free(data->documents[i].feature_indices);
        if (data->documents[i].feature_values) free(data->documents[i].feature_values);
    }
    free(data->documents);
    free(data);
}

void cleanup_vocab(HashTable *ht) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        HashNode *current = ht->table[i];
        while (current != NULL) {
            HashNode *to_free = current;
            current = current->next;
            free(to_free);
        }
    }
}

void cleanup_model(MultinomialNaiveBayes *model) {
    // No dynamic memory in the fixed-size model arrays.
}

// --- Main function for demonstration ---
int main() {
    // 1. Data Loading and Preprocessing
    Dataset *train_data = read_csv_data("train.csv");
    HashTable vocab = {0}; // Initialize Hash Table
    
    // 2. Feature Extraction (Bag-of-Words and TF)
    build_vocabulary(train_data, &vocab);
    calculate_tfidf(train_data, &vocab); // Conceptual call
    
    // 3. Classification Training
    MultinomialNaiveBayes model = {0};
    // This step also runs calculate_tf on all documents
    train_naive_bayes(train_data, &vocab, &model);
    
    // 4. Prediction
    const char *test_article_real = "The president gave a big speech in the city.";
    int prediction_real = predict_naive_bayes(test_article_real, &vocab, &model);
    
    const char *test_article_fake = "New York fact: the best amazing city is totally fake.";
    int prediction_fake = predict_naive_bayes(test_article_fake, &vocab, &model);
    
    printf("\n====================================\n");
    printf("        FINAL PREDICTION RESULTS    \n");
    printf("====================================\n");
    printf("Test Article 1 (Real-leaning): \"%s\"\n", test_article_real);
    printf("-> Predicted Label: %s\n", (prediction_real == REAL_CLASS ? "REAL" : "FAKE"));
        
    printf("Test Article 2 (Fake-leaning): \"%s\"\n", test_article_fake);
    printf("-> Predicted Label: %s\n", (prediction_fake == REAL_CLASS ? "REAL" : "FAKE"));

    // 5. Cleanup
    cleanup_dataset(train_data);
    cleanup_vocab(&vocab);
    cleanup_model(&model);
    
    return 0;
}