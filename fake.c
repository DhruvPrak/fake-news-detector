#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <ctype.h> 

#define MAX_WORD_LEN 50
#define VOCAB_SIZE 20000 // Hash Table size
#define MAX_FEATURES 10000
#define NUM_CLASSES 2 // Fake and Real
#define REAL_CLASS 0
#define FAKE_CLASS 1

typedef struct HashNode {
    char word[MAX_WORD_LEN];
    int word_id;
    int count;
    struct HashNode *next;
} HashNode;

typedef struct {
    HashNode *table[VOCAB_SIZE];
    int total_words; 
} HashTable;

typedef struct {
    int id;
    int label;
    char *text; 
    int *feature_indices; 
    double *feature_values; 
    int feature_count;
} Document;

typedef struct {
    Document *documents;
    int size;
    int capacity;
} Dataset;

typedef struct SparseNode {
    int doc_id;
    int feature_id;
    double value;
    struct SparseNode *next_in_row;
    struct SparseNode *next_in_col;
} SparseNode;


typedef struct {
    double log_prior[NUM_CLASSES];          
    double log_likelihood[VOCAB_SIZE][NUM_CLASSES]; 
    long long class_total_word_count[NUM_CLASSES]; 
    long long global_total_word_count;     
    int vocabulary_size;      
} MultinomialNaiveBayes;

unsigned int hash(const char *str);
HashNode *ht_insert(HashTable *ht, const char *word);
void tokenize(const char *text, HashTable *vocab, Document *doc);
void build_vocabulary(Dataset *data, HashTable *vocab);
Dataset *read_csv_data(const char *filepath);
void calculate_tf(Document *doc, HashTable *vocab);
void calculate_tfidf(Dataset *data, HashTable *vocab);
void train_naive_bayes(Dataset *train_data, HashTable *vocab, MultinomialNaiveBayes *model);
int predict_naive_bayes(const char *text, HashTable *vocab, MultinomialNaiveBayes *model);
void cleanup_dataset(Dataset *data);
void cleanup_vocab(HashTable *ht);
void cleanup_model(MultinomialNaiveBayes *model);

unsigned int hash(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % VOCAB_SIZE;
}

HashNode *ht_insert(HashTable *ht, const char *word) {
    unsigned int index = hash(word);
    HashNode *current = ht->table[index];

    while (current != NULL) {
        if (strcmp(current->word, word) == 0) {
            current->count++;
            return current;
        }
        current = current->next;
    }

    if (ht->total_words >= VOCAB_SIZE) {
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
    new_node->word_id = ht->total_words++;
    new_node->next = ht->table[index];
    ht->table[index] = new_node;

    return new_node;
}

Dataset *read_csv_data(const char *filepath) {
    printf("--- NOTE: Conceptual CSV Reader (File: %s) ---\n", filepath);
    printf("Using a small hardcoded dataset for demonstration of the ML flow.\n");
    
    Dataset *data = (Dataset *)malloc(sizeof(Dataset));
    data->size = 4;
    data->capacity = 4;
    data->documents = (Document *)malloc(data->capacity * sizeof(Document));
    
    data->documents[0].id = 0;
    data->documents[0].label = REAL_CLASS;
    data->documents[0].text = strdup("The president visited New York and held a meeting.");
    data->documents[0].feature_indices = NULL; data->documents[0].feature_values = NULL; data->documents[0].feature_count = 0;

    data->documents[1].id = 1;
    data->documents[1].label = FAKE_CLASS;
    data->documents[1].text = strdup("Amazing fact: New York is the best city to live.");
    data->documents[1].feature_indices = NULL; data->documents[1].feature_values = NULL; data->documents[1].feature_count = 0;
    
    data->documents[2].id = 2;
    data->documents[2].label = REAL_CLASS;
    data->documents[2].text = strdup("The president gave a speech about the economy.");
    data->documents[2].feature_indices = NULL; data->documents[2].feature_values = NULL; data->documents[2].feature_count = 0;
    
    data->documents[3].id = 3;
    data->documents[3].label = FAKE_CLASS;
    data->documents[3].text = strdup("This city is totally fake and the best city.");
    data->documents[3].feature_indices = NULL; data->documents[3].feature_values = NULL; data->documents[3].feature_count = 0;

    printf("Loaded %d sample documents for demonstration.\n", data->size);
    return data;
}

void tokenize(const char *text, HashTable *vocab, Document *doc) {
    char *text_copy = strdup(text);
    char *saveptr = text_copy;
    char *token;
    
    char delimiters[] = " .,!?\"'():-;"; 

    int current_capacity = 20;
    int current_count = 0;
    int *temp_indices = (int *)malloc(current_capacity * sizeof(int));
    
    token = strtok(text_copy, delimiters); 
    
    while (token != NULL) {
        for (int i = 0; token[i]; i++) {
            token[i] = tolower((unsigned char) token[i]);
        }
        if (strlen(token) > 2) {
            HashNode *node = ht_insert(vocab, token);
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
    
    free(saveptr);
    
    doc->feature_indices = temp_indices;
    doc->feature_count = current_count;
    doc->feature_indices = (int *)realloc(doc->feature_indices, current_count * sizeof(int));
}

void build_vocabulary(Dataset *data, HashTable *vocab) {
    printf("\n--- Building Vocabulary and Tokenizing Documents ---\n");
    for (int i = 0; i < data->size; i++) {
        tokenize(data->documents[i].text, vocab, &data->documents[i]);
    }
    printf("Vocabulary built with %d unique words.\n", vocab->total_words);
}
void calculate_tf(Document *doc, HashTable *vocab) {
    if (doc->feature_count == 0) return;

    doc->feature_values = (double *)calloc(vocab->total_words, sizeof(double));
    if (doc->feature_values == NULL) {
        perror("Calloc failed for TF array");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < doc->feature_count; i++) {
        int word_id = doc->feature_indices[i];
        if (word_id < vocab->total_words) {
            doc->feature_values[word_id]++; 
        }
    }
    doc->feature_count = vocab->total_words; 
}

void calculate_tfidf(Dataset *data, HashTable *vocab) {
    printf("--- NOTE: Conceptual TF-IDF/Sparse Matrix Function Called ---\n");
}

void train_naive_bayes(Dataset *train_data, HashTable *vocab, MultinomialNaiveBayes *model) {
    printf("\n--- Training Multinomial Naive Bayes Model ---\n");
    long long (*class_word_counts)[VOCAB_SIZE] = calloc(NUM_CLASSES, sizeof(*class_word_counts));
    if (class_word_counts == NULL) {
        perror("Failed to allocate class_word_counts");
        exit(EXIT_FAILURE);
    }
    
    model->global_total_word_count = 0;
    model->vocabulary_size = vocab->total_words;
    int class_document_count[NUM_CLASSES] = {0};
    
    for (int i = 0; i < train_data->size; i++) {
        Document *doc = &train_data->documents[i];
        int c = doc->label;
        class_document_count[c]++;
        calculate_tf(doc, vocab);
        for (int j = 0; j < doc->feature_count; j++) {
            int count = (int)doc->feature_values[j];
            
            if (count > 0) {
                class_word_counts[c][j] += count;
                model->class_total_word_count[c] += count;
            }
        }
    }
    
    model->global_total_word_count = model->class_total_word_count[REAL_CLASS] + model->class_total_word_count[FAKE_CLASS];

    for (int c = 0; c < NUM_CLASSES; c++) {
        model->log_prior[c] = log((double)(class_document_count[c] + 1) / (train_data->size + NUM_CLASSES));
    }
    for (int c = 0; c < NUM_CLASSES; c++) {
        double denominator = (double)model->class_total_word_count[c] + model->vocabulary_size;
        
        for (int w = 0; w < model->vocabulary_size; w++) {
            double numerator = (double)class_word_counts[c][w] + 1.0;
            
            model->log_likelihood[w][c] = log(numerator / denominator);
        }
    }

    free(class_word_counts);
    printf("Training complete. Vocabulary size: %d.\n", model->vocabulary_size);
}

int predict_naive_bayes(const char *text, HashTable *vocab, MultinomialNaiveBayes *model) {
    Document new_doc = {0};
    tokenize(text, vocab, &new_doc);
    double log_prob[NUM_CLASSES] = {0.0};
    for (int c = 0; c < NUM_CLASSES; c++) {
        log_prob[c] = model->log_prior[c];
        for (int i = 0; i < new_doc.feature_count; i++) {
            int word_id = new_doc.feature_indices[i];
            if (word_id < model->vocabulary_size) {
                log_prob[c] += model->log_likelihood[word_id][c];
            }
        }
    }

    if (new_doc.feature_indices) free(new_doc.feature_indices);
    if (log_prob[FAKE_CLASS] > log_prob[REAL_CLASS]) {
        return FAKE_CLASS;
    } else {
        return REAL_CLASS;
    }
}

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
}

int main() {
    Dataset *train_data = read_csv_data("train.csv");
    HashTable vocab = {0};
    build_vocabulary(train_data, &vocab);
    calculate_tfidf(train_data, &vocab);
    MultinomialNaiveBayes model = {0};
    train_naive_bayes(train_data, &vocab, &model);
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
    cleanup_dataset(train_data);
    cleanup_vocab(&vocab);
    cleanup_model(&model);
    
    return 0;

}
