/*
 * GUI LLAMA2.C
 * 
 * BASED UPON LLAMA2.C by Andrej
 * GUI/Dreamcast port by gameblabla
 * 
 * LICENSE MIT
 * 
 * */
#include <SDL/SDL.h>
#ifdef DREAMCAST
#include <SDL/SDL_dreamcast.h>
#include <dc/maple.h>
#include <dc/maple/controller.h>
#include <arch/cache.h>
#endif
#include <stdio.h>

#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdarg.h>


#include "dc.h"

// We assume you have a "font_drawing.h" that declares print_string(...).
// That function draws an 8×8 font onto a 16-bpp buffer at (x, y) with fg/bg.
#include "font_drawing.h"

/* ========================================================================== */
/*                      Llama2 Inference Code (unchanged)                     */
/* ========================================================================== */

#define NO_SIMD
#define NO_FAST_MATH
//#define LEGACY_FP

typedef unsigned long long uint64;
#define REAL_EPSILON 1e-5f

#ifdef LEGACY_FP
static float safe_exp(float x) {
  // Avoid overflow
  if (x > 88.0) return 1e38;
  if (x < -88.0) return 0;
  return exp(x);
}
static float safe_sqrt(float x) {
  if (x <= 0) return 0;
  return sqrt(x);
}
#else
#define safe_exp expf
#define safe_sqrt SQRTF_REAL
#endif

static void log_debug(const char* format, ...) {
    static FILE* debug_file = NULL;
    va_list args;
    if (!debug_file) {
        debug_file = fopen("debug.log", "w");
        if (!debug_file) return;
    }
    va_start(args, format);
    vfprintf(debug_file, format, args);
    va_end(args);
    fflush(debug_file);
}

// Helper function to read floats from file
static void read_weights_from_file(FILE* file, float* dest, size_t n_elements) {
  float* temp;
  size_t i, chunk_size, current_chunk, remaining;
  const size_t MAX_CHUNK = 16384; // 16K elements

  if (sizeof(float) == sizeof(float)) {
    // Direct read
    remaining = n_elements;
    while (remaining > 0) {
      chunk_size = (remaining < MAX_CHUNK) ? remaining : MAX_CHUNK;
      if (fread(dest, sizeof(float), chunk_size, file) != chunk_size) {
        log_debug("Failed to read chunk of %ld elements\n", (long)chunk_size);
        exit(EXIT_FAILURE);
      }
      dest += chunk_size;
      remaining -= chunk_size;
    }
  } else {
    // Fallback for unusual systems
    temp = (float*)malloc(MAX_CHUNK * sizeof(float));
    if (!temp) {
      log_debug("Failed to allocate temp buffer\n");
      exit(EXIT_FAILURE);
    }

    remaining = n_elements;
    while (remaining > 0) {
      current_chunk = (remaining < MAX_CHUNK) ? remaining : MAX_CHUNK;
      if (fread(temp, sizeof(float), current_chunk, file) != current_chunk) {
        log_debug("Failed to read chunk\n");
        free(temp);
        exit(EXIT_FAILURE);
      }
      for (i = 0; i < current_chunk; i++) {
        dest[i] = (float)temp[i];
      }
      dest += current_chunk;
      remaining -= current_chunk;
    }
    free(temp);
  }
}

/* ========================================================================== */
/*                      Transformer + Tokenizer + Sampler                     */
/* ========================================================================== */

typedef struct {
    int dim;        
    int hidden_dim; 
    int n_layers;   
    int n_heads;    
    int n_kv_heads; 
    int vocab_size; 
    int seq_len;    
} Config;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} TransformerWeights;

typedef struct {
    float *x;      
    float *xb;     
    float *xb2;    
    float *hb;     
    float *hb2;    
    float *q;      
    float *k;      
    float *v;      
    float *att;    
    float *logits; 

    float* key_cache;   
    float* value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x    = calloc(p->dim,        sizeof(float));
    s->xb   = calloc(p->dim,        sizeof(float));
    s->xb2  = calloc(p->dim,        sizeof(float));
    s->hb   = calloc(p->hidden_dim, sizeof(float));
    s->hb2  = calloc(p->hidden_dim, sizeof(float));
    s->q    = calloc(p->dim,        sizeof(float));
    s->att  = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache   = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    if(!s->x||!s->xb||!s->xb2||!s->hb||!s->hb2||!s->q||
       !s->att||!s->logits||!s->key_cache||!s->value_cache){
        fprintf(stderr,"Failed malloc run_state\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void read_checkpoint(char* checkpoint, Config* c, TransformerWeights* w) {
    FILE *f = fopen(checkpoint,"rb");
    if(!f){ fprintf(stderr,"Cannot open checkpoint\n"); exit(1); }
    if(fread(c,sizeof(Config),1,f)!=1){
        fprintf(stderr,"Failed reading config\n");
        exit(1);
    }
    int shared = c->vocab_size>0 ? 1 : 0;
    c->vocab_size = abs(c->vocab_size);
    int head_size= c->dim/c->n_heads;
    // read tables
    size_t offset= sizeof(Config);
    size_t emb_sz= c->vocab_size*(size_t)c->dim;
    w->token_embedding_table= malloc(emb_sz*sizeof(float));
    fseek(f, offset, SEEK_SET);
    offset+= emb_sz*sizeof(float);
    if(!w->token_embedding_table)exit(1);
    if(fread(w->token_embedding_table,sizeof(float),emb_sz,f)!=emb_sz){
        fprintf(stderr,"Failed read token_embed\n");
        exit(1);
    }
    size_t rms_sz= c->n_layers*(size_t)c->dim;
    w->rms_att_weight= malloc(rms_sz*sizeof(float));
    read_weights_from_file(f,w->rms_att_weight,rms_sz);
    offset+= rms_sz*sizeof(float);

    size_t mat_sz= c->dim*(size_t)c->n_heads*head_size;
    w->wq= malloc(c->n_layers*mat_sz*sizeof(float));
    read_weights_from_file(f,w->wq,c->n_layers*mat_sz);
    offset+= c->n_layers*mat_sz*sizeof(float);

    mat_sz= c->dim*(size_t)c->n_kv_heads*head_size;
    w->wk= malloc(c->n_layers*mat_sz*sizeof(float));
    read_weights_from_file(f,w->wk,c->n_layers*mat_sz);
    offset+= c->n_layers*mat_sz*sizeof(float);

    w->wv= malloc(c->n_layers*mat_sz*sizeof(float));
    read_weights_from_file(f,w->wv,c->n_layers*mat_sz);
    offset+= c->n_layers*mat_sz*sizeof(float);

    mat_sz= c->n_heads*(size_t)head_size*c->dim;
    w->wo= malloc(c->n_layers*mat_sz*sizeof(float));
    read_weights_from_file(f,w->wo,c->n_layers*mat_sz);
    offset+= c->n_layers*mat_sz*sizeof(float);

    w->rms_ffn_weight= malloc(rms_sz*sizeof(float));
    read_weights_from_file(f,w->rms_ffn_weight,rms_sz);
    offset+= rms_sz*sizeof(float);

    size_t layer_sz= c->dim*(size_t)c->hidden_dim;
    w->w1= malloc(c->n_layers*layer_sz*sizeof(float));
    read_weights_from_file(f,w->w1,c->n_layers*layer_sz);
    offset+= c->n_layers*layer_sz*sizeof(float);

    w->w2= malloc(c->n_layers*layer_sz*sizeof(float));
    w->w3= malloc(c->n_layers*layer_sz*sizeof(float));
    read_weights_from_file(f,w->w2,c->n_layers*layer_sz);
    offset+= c->n_layers*layer_sz*sizeof(float);

    read_weights_from_file(f,w->w3,c->n_layers*layer_sz);
    offset+= c->n_layers*layer_sz*sizeof(float);

    w->rms_final_weight= malloc(c->dim*sizeof(float));
    read_weights_from_file(f,w->rms_final_weight,c->dim);
    offset+= c->dim*sizeof(float);

    offset+= c->seq_len*head_size*sizeof(float);

    if(!shared){
        w->wcls= malloc(c->vocab_size*(size_t)c->dim*sizeof(float));
        read_weights_from_file(f,w->wcls,c->vocab_size*(size_t)c->dim);
    } else {
        w->wcls= w->token_embedding_table;
    }
    fclose(f);
}

void build_transformer(Transformer* t, char* cp) {
    read_checkpoint(cp, &t->config, &t->weights);
    malloc_run_state(&t->state, &t->config);
}

static void free_weights(TransformerWeights* w){
    if(w->token_embedding_table) free(w->token_embedding_table);
    if(w->rms_att_weight)        free(w->rms_att_weight);
    if(w->wq)                    free(w->wq);
    if(w->wk)                    free(w->wk);
    if(w->wv)                    free(w->wv);
    if(w->wo)                    free(w->wo);
    if(w->rms_ffn_weight)        free(w->rms_ffn_weight);
    if(w->w1)                    free(w->w1);
    if(w->w2)                    free(w->w2);
    if(w->w3)                    free(w->w3);
    if(w->rms_final_weight)      free(w->rms_final_weight);
    if(w->wcls && w->wcls!= w->token_embedding_table){
        free(w->wcls);
    }
}

void free_transformer(Transformer* t){
    free_weights(&t->weights);
    free_run_state(&t->state);
}

void rmsnorm(float* o,float* x,float*w,int size){
    float ss=0.f;
    for(int j=0;j<size;j++)
    { 
		//ss+= x[j]*x[j]; 
		ss = FMAC(x[j], x[j], ss); 
	}
    ss/= size;
    //ss = DIVIDE_REAL(size, ss);
    
    ss+= REAL_EPSILON;
    //float scale= 1.f/sqrtf(ss);
    float scale = FSSRA_REAL(ss);
    
    for(int j=0;j<size;j++){
        o[j]= x[j]* scale * w[j];
    }
}
void softmax(float*x,int size){
    float mx= x[0];
    for(int i=1;i<size;i++){
        if(x[i]>mx) mx=x[i];
    }
    float sum=0.f;
    for(int i=0;i<size;i++){
        x[i]= expf(x[i]- mx);
        sum+= x[i];
    }
    for(int i=0;i<size;i++){
        x[i]/= sum;
    }
}

#define ALIGN16 __attribute__((aligned(16)))

#define PREFETCH(addr) __builtin_prefetch(addr)

void matmul( float *xout,  const float *x,  const float *w, int n, int d) {
    int i = 0;
    for (int i = 0; i < d; i++) {
        float val0 = 0.0f;
        float val1 = 0.0f;
        float val2 = 0.0f;
        float val3 = 0.0f;

        size_t off = (size_t)i * n;

        dcache_alloc_block(&xout[i], i);

        int j;
        for (j = 0; j <= n - 4; j += 4) {
            __builtin_prefetch(&x[j + 8]);
            __builtin_prefetch(&w[off + j + 8]);

            // Perform FMAC operations
            val0 = fmaf(x[j],     w[off + j],     val0);
            val1 = fmaf(x[j + 1], w[off + j + 1], val1);
            val2 = fmaf(x[j + 2], w[off + j + 2], val2);
            val3 = fmaf(x[j + 3], w[off + j + 3], val3);
        }

        // Handle remaining elements
        for (; j < n; j++) {
            val0 = fmaf(x[j], w[off + j], val0);
        }

        // Sum the accumulated values
        float sum = val0 + val1 + val2 + val3;

        // Store the result to xout[i]
        xout[i] = sum;
    }
}


float* forward(Transformer* t,int token,int pos){
    Config* p= &t->config;
    TransformerWeights* w= &t->weights;
    RunState* s= &t->state;
    float* x= s->x;
    int dim= p->dim;
    int kv_dim= DIVIDE_REAL((p->dim*p->n_kv_heads), p->n_heads);
    int kv_mul= DIVIDE_REAL(p->n_heads, p->n_kv_heads);
    int hidden_dim= p->hidden_dim;
    int head_size= DIVIDE_REAL(dim,p->n_heads);

    // embedding
    float* embedRow= w->token_embedding_table + token*(size_t)dim;
    MEMCPY_REAL(x, embedRow, dim*sizeof(*x));

    for(int l=0;l<p->n_layers;l++){
        // att norm
        rmsnorm(s->xb, x, w->rms_att_weight + l*(size_t)dim, dim);
        int loff= l*p->seq_len* kv_dim;
        s->k= s->key_cache + loff + pos*kv_dim;
        s->v= s->value_cache + loff + pos*kv_dim;

        size_t matOff= l*(size_t)dim*dim;
        matmul(s->q, s->xb, w->wq+ matOff, dim, dim);

        matOff= l*(size_t)dim*kv_dim;
        matmul(s->k, s->xb, w->wk+ matOff, dim, kv_dim);
        matmul(s->v, s->xb, w->wv+ matOff, dim, kv_dim);

        // rope
        for(int i=0;i<dim;i+=2){
            int hd= i% head_size;
            float freq= 1.0f/ powf(10000.f, DIVIDE_REAL(hd,(float)head_size));
            float val= pos*freq;
            float c= cosf(val), s2= sinf(val);
            if(i< kv_dim){
                float k0=s->k[i], k1=s->k[i+1];
                s->k[i]= k0*c - k1*s2;
                s->k[i+1]= k0*s2 + k1*c;
            }
            {
                float q0=s->q[i], q1=s->q[i+1];
                s->q[i]= q0*c - q1*s2;
                s->q[i+1]= q0*s2 + q1*c;
            }
        }

        // att heads
        for(int h=0;h< p->n_heads;h++){
            float*qq= s->q + h*head_size;
            float*att= s->att+ h*p->seq_len;
            for(int t=0;t<=pos;t++){
                float*kk= s->key_cache + loff + t*kv_dim + (h/kv_mul)* head_size;
                float sc=0.f;
                for(int i=0;i< head_size;i++){
					sc = FMAC(kk[i], qq[i], sc);
                    //sc+= qq[i]* kk[i];
                }
                sc/= SQRTF_REAL(head_size);
                att[t]= sc;
            }
            softmax(att, pos+1);
            float*xb= s->xb+ h*head_size;
            MEMSET_REAL(xb,0, head_size*sizeof(float));
            for(int t=0;t<=pos;t++){
                float*vv= s->value_cache + loff + t*kv_dim + (h/kv_mul)* head_size;
                float a= att[t];
                for(int i=0;i< head_size;i++){
                    //xb[i]+= a* vv[i];
                    xb[i] = FMAC(vv[i],a,xb[i]);
                }
            }
        }
        // wo
        matOff= l*(size_t)dim*dim;
        matmul(s->xb2, s->xb, w->wo+ matOff, dim, dim);
        for(int i=0;i<dim;i++){
            x[i]+= s->xb2[i];
        }
        // ffn
        rmsnorm(s->xb, x, w->rms_ffn_weight+ l*(size_t)dim, dim);
        matOff= l*(size_t)dim* hidden_dim;
        matmul(s->hb, s->xb, w->w1+ matOff, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3+ matOff, dim, hidden_dim);
        for(int i=0;i< hidden_dim;i++){
            float v2= s->hb[i];
            v2*= (1.f/(1.f+ expf(-v2)));
            v2*= s->hb2[i];
            s->hb[i]= v2;
        }
        matmul(s->xb, s->hb, w->w2+ matOff, hidden_dim, dim);
        for(int i=0;i<dim;i++){
            x[i]+= s->xb[i];
        }
    }
    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

/* BPE Tokenizer  (brief) */
typedef struct {
    char* str;
    int   id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void*a,const void*b){
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t,char* path,int vsz){
    t->vocab_size= vsz;
    t->vocab= malloc(vsz*sizeof(char*));
    t->vocab_scores= malloc(vsz*sizeof(float));
    t->sorted_vocab= NULL;
    FILE* f= fopen(path,"rb");
    if(!f){ fprintf(stderr,"cannot open tok file\n"); exit(1); }
    unsigned char buf[32];
    fread(buf,1,32,f); // skip
    fseek(f,0,SEEK_SET);
    fseek(f,0,SEEK_END);
    long fs= ftell(f);
    fseek(f,0,SEEK_SET);
    if(fread(&t->max_token_length,sizeof(int),1,f)!=1){
        fprintf(stderr,"tok read fail\n");
        exit(1);
    }
    for(int i=0;i<256;i++){
        t->byte_pieces[i*2]= (unsigned char)i;
        t->byte_pieces[i*2+1]= '\0';
    }
    for(int i=0;i<vsz;i++){
        float sc;
        fread(&sc,sizeof(float),1,f);
        t->vocab_scores[i]= sc;
        unsigned char lb[4];
        fread(lb,1,4,f);
        int len= (lb[3]<<24)|(lb[2]<<16)|(lb[1]<<8)|lb[0];
        t->vocab[i]= malloc(len+1);
        fread(t->vocab[i], len,1,f);
        t->vocab[i][len]='\0';
    }
    fclose(f);
}

void free_tokenizer(Tokenizer* t){
    if(!t)return;
    for(int i=0;i<t->vocab_size;i++){
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t,int prev,int tok){
    char* piece= t->vocab[tok];
    if(prev==1 && piece[0]==' ') piece++;
    unsigned char bv;
    if(sscanf(piece,"<0x%02hhX>",&bv)==1){
        piece= (char*)(t->byte_pieces + bv*2);
    }
    return piece;
}

static void safe_str_append(char*out,size_t outsz,const char*piece){
    if(!piece||!piece[0])return;
    if(piece[1]=='\0'){
        unsigned char bv= piece[0];
        if(isprint(bv)||isspace(bv)){
            strncat(out, piece,outsz- strlen(out)-1);
        }
    } else {
        strncat(out, piece,outsz- strlen(out)-1);
    }
}

int str_lookup(char* str, TokenIndex* sv,int vsz){
    TokenIndex tmp; tmp.str=str; tmp.id=0;
    TokenIndex* r= bsearch(&tmp, sv, vsz,sizeof(TokenIndex), compare_tokens);
    return r? r->id:-1;
}

void encode(Tokenizer* t,char* txt,int8_t bos,int8_t eos, int*tokens,int*n_tokens){
    if(!txt){ fprintf(stderr,"encode null\n"); exit(1);}
    if(!t->sorted_vocab){
        t->sorted_vocab= malloc(t->vocab_size*sizeof(TokenIndex));
        for(int i=0;i<t->vocab_size;i++){
            t->sorted_vocab[i].str= t->vocab[i];
            t->sorted_vocab[i].id= i;
        }
        qsort(t->sorted_vocab,t->vocab_size,sizeof(TokenIndex), compare_tokens);
    }
    char*sbuf= malloc(t->max_token_length*2+10);
    size_t sl=0; *n_tokens=0;
    if(bos) tokens[(*n_tokens)++]=1;
    if(txt[0]){
        int idx= str_lookup(" ", t->sorted_vocab,t->vocab_size);
        tokens[(*n_tokens)++]= idx;
    }
    for(char*c=txt; *c;c++){
        if((*c &0xC0)!=0x80) sl=0;
        sbuf[sl++]= *c; sbuf[sl]='\0';
        if((*(c+1)&0xC0)==0x80 && sl<4) continue;
        int id= str_lookup(sbuf,t->sorted_vocab,t->vocab_size);
        if(id!=-1){
            tokens[(*n_tokens)++]= id;
        } else {
            for(int ii=0;ii<(int)sl;ii++){
                tokens[(*n_tokens)++]= (unsigned char)sbuf[ii]+3;
            }
        }
        sl=0;
    }
    while(1){
        float best=-1e10f; int bestid=-1,besti=-1;
        for(int i=0;i<(*n_tokens-1);i++){
            sprintf(sbuf,"%s%s",t->vocab[tokens[i]],t->vocab[tokens[i+1]]);
            int idx= str_lookup(sbuf,t->sorted_vocab,t->vocab_size);
            if(idx!=-1 && t->vocab_scores[idx]>best){
                best= t->vocab_scores[idx];
                bestid= idx; besti=i;
            }
        }
        if(besti==-1)break;
        tokens[besti]= bestid;
        for(int i=besti+1;i<(*n_tokens-1);i++){
            tokens[i]= tokens[i+1];
        }
        (*n_tokens)--;
    }
    if(eos) tokens[(*n_tokens)++]=2;
    free(sbuf);
}

/* Sampler */
typedef struct {
    float prob;
    int   index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    uint64 rng_state;
} Sampler;

int sample_argmax(float*p,int n){
    float mx= p[0]; int mi=0;
    for(int i=1;i<n;i++){
        if(p[i]>mx){ mx=p[i]; mi=i;}
    }
    return mi;
}

int sample_mult(float*p,int n,float c){
    float cdf=0;
    for(int i=0;i<n;i++){
        cdf+= p[i];
        if(c< cdf)return i;
    }
    return n-1;
}

static int cmp_pi(const void*a,const void*b){
    ProbIndex*aa=(ProbIndex*)a; ProbIndex*bb=(ProbIndex*)b;
    if(aa->prob>bb->prob)return -1;
    if(aa->prob<bb->prob)return 1;
    return 0;
}

int sample_topp(float*p,int n,float top, ProbIndex*pi,float coin){
    int n0=0;
    float cutoff= (1.f- top)/(n-1);
    for(int i=0;i<n;i++){
        if(p[i]>= cutoff){
            pi[n0].prob= p[i];
            pi[n0].index= i;
            n0++;
        }
    }
    qsort(pi,n0,sizeof(ProbIndex), cmp_pi);
    float total=0; int last=n0-1;
    for(int i=0;i<n0;i++){
        total+= pi[i].prob;
        if(total> top){ last=i;break;}
    }
    float r= coin* total, cdf=0;
    for(int i=0;i<= last;i++){
        cdf+= pi[i].prob;
        if(r< cdf) return pi[i].index;
    }
    return pi[last].index;
}

void build_sampler(Sampler*s,int vsz,float temp,float topp,uint64 sd){
    s->vocab_size= vsz; s->temperature= temp; s->topp= topp; s->rng_state= sd;
    s->probindex= malloc(vsz*sizeof(ProbIndex));
}

void free_sampler(Sampler*s){
    if(s->probindex) free(s->probindex);
    s->probindex=NULL;
}

unsigned int random_u32(uint64*st){
    *st^= *st>>12; *st^= *st<<25; *st^= *st>>27;
    return (unsigned int)((*st*(uint64)0x2545F4914F6CDD1D)>>32);
}
float random_f32(uint64*st){
    return (random_u32(st)>>8)/16777216.f;
}

int sample(Sampler*s, float* logits){
    int nx;
    if(s->temperature==0.f){
        nx= sample_argmax(logits, s->vocab_size);
    } else {
        for(int i=0;i<s->vocab_size;i++){
            DIVIDE_REAL(s->temperature,logits[i]);
        }
        softmax(logits, s->vocab_size);
        float c= random_f32(&s->rng_state);
        if(s->topp<=0.f || s->topp>=1.f){
            nx= sample_mult(logits, s->vocab_size, c);
        } else {
            nx= sample_topp(logits, s->vocab_size, s->topp, s->probindex, c);
        }
    }
    return nx;
}

static inline long time_in_ms(){
    struct timespec tm; clock_gettime(CLOCK_REALTIME,&tm);
    return tm.tv_sec*1000 + tm.tv_nsec/1000000;
}

void generate(Transformer*tx,Tokenizer*tz,Sampler*sm,
              char* prompt,int steps,char*outbuf,size_t outsz){
    if(!prompt) prompt="";
    int npt=0; 
    int* ptoks= malloc((strlen(prompt)+4)*sizeof(int));
    encode(tz, prompt, 1,0, ptoks,&npt);
    int token= ptoks[0];
    int pos=0;
    snprintf(outbuf, outsz, "");

    long st=0,en=0;
    while(pos<steps){
        float*lg= forward(tx, token,pos);
        int nx;
        if(pos< npt-1){
            nx= ptoks[pos+1];
        } else {
            nx= sample(sm, lg);
        }
        pos++;
        if(nx==1) break;
        char* piece= decode(tz, token,nx);
        if(piece && piece[0]){
            if(piece[1]=='\0'){
                unsigned char bv= piece[0];
                if(isprint(bv)||isspace(bv)){
                    strncat(outbuf,piece,outsz- strlen(outbuf)-1);
                }
            } else {
                strncat(outbuf,piece,outsz- strlen(outbuf)-1);
            }
        }
        token= nx;
        if(st==0) st= time_in_ms();
    }
	if(pos>1){
        en= time_in_ms();
        float rate= DIVIDE_REAL((pos-1), ((float)(en-st)/1000.f));
        char sbuf[64]; sprintf(sbuf,"\n(%.2f tokens/s)\n", rate);
        strncat(outbuf,sbuf,outsz- strlen(outbuf)-1);
    }
    free(ptoks);
}

/******************************************************************************
 * Now the main SDL UI in 16bpp with print_string() 
 *****************************************************************************/

#define SCREEN_WIDTH  640
#define SCREEN_HEIGHT 480
#define SCREEN_BPP    16

SDL_Surface* gScreen= NULL;
static SDL_Joystick*gJoystick= NULL;

// Two screens: 0=input, 1=output
static int gUIState= 0;

// We'll define the "virtual keyboard" layout with a single "[SPACE]" row
static const char*gVirtualKB[]={
  "Q W E R T Y U I O P",
  "A S D F G H J K L [BKSP]",
  "Z X C V B N M , . [ENTER]",
  "[SPACE]"
};
static const int KB_ROWS=4;
static const int KB_ROW_HEIGHT=20;
static const int KB_MARGIN_TOP=360;
static const int KB_COL_WIDTH=40;

static int kb_cursor_row=0;
static int kb_cursor_col=0;
static int gAxisSign[2]={0,0};


// The user prompt + output
static char gUserInput[1024];
static char gModelOutput[8192];

// We'll keep references to our Llama objects
static Transformer gTransformer;
static Tokenizer   gTokenizer;
static Sampler     gSampler;

// draws a single line of text using print_string
static void draw_text_line(const char* msg, int x, int y, uint16_t fg, uint16_t bg) {
    uint16_t* pixels= (uint16_t*)gScreen->pixels;
    print_string(msg, fg, bg, x, y, pixels);
}

// A naive "wrapped text" version
static void draw_text_wrapped(const char* text, int x, int y, int wrapWidth,
                              uint16_t fg, uint16_t bg)
{
    char copy[8192];
    strncpy(copy, text, sizeof(copy));
    copy[8191]='\0';
    char* token= strtok(copy, " \n");
    int curX= x, curY= y;
    int charW= 8; // each char 8 px wide
    while(token){
        int tokenLen= strlen(token);
        int tokenPx= tokenLen* charW;
        int spacePx= charW;
        if(curX+ tokenPx+ spacePx > x+ wrapWidth){
            curY+=16;
            curX= x;
        }
        // build "token + space"
        char buffer[256];
        snprintf(buffer,sizeof(buffer),"%s ", token);
        print_string(buffer, fg, bg, curX, curY, (uint16_t*)gScreen->pixels);
        curX += tokenPx+ spacePx;
        token= strtok(NULL," \n");
    }
}

static void render_input_screen(void) {
    SDL_FillRect(gScreen,NULL,0);
    
    draw_text_line("llama2.c Dreamcast port by Gameblabla, 2024",20,20, TextWhite,0);
    draw_text_line("Your Input:",20,40, TextWhite,0);
    draw_text_line(gUserInput,20,60, TextWhite,0);

    // draw keyboard
    int y= KB_MARGIN_TOP;
    for(int row=0; row<KB_ROWS; row++){
        char rowcpy[128];
        strncpy(rowcpy,gVirtualKB[row],sizeof(rowcpy));
        rowcpy[127]='\0';
        char* ctx=NULL;
        char* tok= strtok(rowcpy," ");
        int col=0;
        int baseX= 20;
        while(tok){
            uint16_t c= TextWhite;
            if(row== kb_cursor_row && col== kb_cursor_col){
                c= TextRed;
            }
            draw_text_line(tok, baseX + col*KB_COL_WIDTH, y, c, 0);
            col++;
            tok= strtok(NULL," ");
        }
        y+= KB_ROW_HEIGHT;
    }
    SDL_Flip(gScreen);
}

static void render_output_screen(void){
    SDL_FillRect(gScreen,NULL,0);
    
    draw_text_wrapped(gModelOutput,20,20,600, TextWhite,0);
    draw_text_line("(Press any key or button to return...)", 20,440, TextBlue,0);
    SDL_Flip(gScreen);
}

static void main_render(void){
    if(gUIState==0)
    {
		render_input_screen();
	}
    else
    {
		render_output_screen();
	}
}

// parse row/col in the keyboard
static const char* keyboard_get_key(int row,int col){
    if(row<0||row>=KB_ROWS)return NULL;
    // if row=3 => single [SPACE], col must be 0
    if(row==3 && col!=0) return NULL;

    char rowcpy[128];
    strncpy(rowcpy,gVirtualKB[row],sizeof(rowcpy));
    rowcpy[127]='\0';
    char* ctx=NULL;
    char* tok= strtok(rowcpy," ");
    int idx=0;
    while(tok){
        if(row==3){
            // row3 => single token => if idx=0 && col=0 => tok
            if(idx==0 && col==0) return tok;
        } else {
            if(idx== col) return tok;
        }
        idx++;
        tok= strtok(NULL," ");
    }
    return NULL;
}

static void LoadingScreen()
{
	SDL_FillRect(gScreen,NULL,0);
	draw_text_line("NOW LOADING, PLEASE WAIT", 20,440, TextWhite,0);
	SDL_Flip(gScreen);
	SDL_FillRect(gScreen,NULL,0);
	draw_text_line("NOW LOADING, PLEASE WAIT", 20,440, TextWhite,0);
	SDL_Flip(gScreen);
}

static void keyboard_select_key(const char* key){
    if(!key) return;
    if(strcmp(key,"[ENTER]")==0){
		
		LoadingScreen();
		
        // do real Llama inference
        generate(&gTransformer, &gTokenizer, &gSampler,
                 gUserInput, 640, gModelOutput,sizeof(gModelOutput));
        MEMSET_REAL(gUserInput,0,sizeof(gUserInput));
        gUIState=1;
    } else if(strcmp(key,"[SPACE]")==0){
        int len= strlen(gUserInput);
        if(len<(int)sizeof(gUserInput)-1){
            gUserInput[len]=' ';
            gUserInput[len+1]='\0';
        }
    } else if(strcmp(key,"[BKSP]")==0){
        int len= strlen(gUserInput);
        if(len>0) gUserInput[len-1]='\0';
    } else {
        // normal text
        int len= strlen(gUserInput);
        int kl= strlen(key);
        if(len+ kl< (int)sizeof(gUserInput)-1){
            strcat(gUserInput, key);
        }
    }
}

int main(int argc, char*argv[]){
#ifdef DREAMCAST
    maple_device_t *cont;
	cont_state_t *state;
    static unsigned int old_dc_buttons = 0;
    char* checkpoint= "/cd/model.bin";
    char* tokenizer= "/cd/tok.bin";
#else
    char* checkpoint= "model.bin";
    char* tokenizer= "tok.bin";
#endif

    float temperature= 1.0f;
    float topp= 0.9f;
    uint64 seed= (uint64)time(NULL);

    // Build Llama2 objects
    build_transformer(&gTransformer, checkpoint);
    build_tokenizer(&gTokenizer, tokenizer, gTransformer.config.vocab_size);
    build_sampler(&gSampler, gTransformer.config.vocab_size, temperature,topp, seed);
    
#ifdef DREAMCAST
    SDL_DC_SetVideoDriver(SDL_DC_DIRECT_VIDEO);
#endif

    // SDL init
    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK)<0){
        fprintf(stderr,"SDL init error: %s\n",SDL_GetError());
        return 1;
    }
    atexit(SDL_Quit);

    gScreen= SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_BPP, SDL_HWSURFACE | SDL_DOUBLEBUF);
    if(!gScreen){
        fprintf(stderr,"SDL_SetVideoMode error:%s\n",SDL_GetError());
        return 1;
    }
    SDL_WM_SetCaption("Llama2 UI - 16bpp + print_string", NULL);

    if(SDL_NumJoysticks()>0){
        gJoystick= SDL_JoystickOpen(0);
        if(gJoystick){
            SDL_JoystickEventState(SDL_ENABLE);
            printf("Opened joystick: %s\n", SDL_JoystickName(0));
        }
    }

    MEMSET_REAL(gUserInput,0,sizeof(gUserInput));
    MEMSET_REAL(gModelOutput,0,sizeof(gModelOutput));

    int running=1;
    while(running){
        SDL_Event e;
        
#ifdef DREAMCAST
        // Poll Dreamcast Maple controller directly
        cont = maple_enum_type(0, MAPLE_FUNC_CONTROLLER);
        if(cont){
            state = (cont_state_t *)maple_dev_status(cont);
            if(state){
                // "pressed" will contain only the bits that have
                // *just now* changed from 0 to 1 (newly pressed).
                unsigned int pressed = (old_dc_buttons ^ state->buttons) & state->buttons;

                if(gUIState==1){
                    // In output screen, any new button press => switch to input
                    if(pressed) {
                        gUIState=0;
                    }
                } else {
                    // In input screen, respond to new button presses only
                    if(pressed & CONT_START) {
                        // Map to ENTER
                        keyboard_select_key("[ENTER]");
                    }
                    if(pressed & CONT_X){
                        // Map to backspace
                        int len= strlen(gUserInput);
                        if(len>0) gUserInput[len-1]='\0';
                    }
                    if(pressed & CONT_B){
                        // Another way to do ENTER
                        keyboard_select_key("[ENTER]");
                    }
                    if(pressed & CONT_A){
                        // "select" => choose current keyboard cell
                        const char* kk= keyboard_get_key(kb_cursor_row,kb_cursor_col);
                        keyboard_select_key(kk);
                    }

                    // D-Pad: each press moves the cursor once
                    if(pressed & CONT_DPAD_UP){
                        if(kb_cursor_row>0){
                            kb_cursor_row--;
                            if(kb_cursor_row==3) kb_cursor_col=0;
                        }
                    }
                    if(pressed & CONT_DPAD_DOWN){
                        kb_cursor_row++;
                        if(kb_cursor_row>=KB_ROWS) kb_cursor_row=KB_ROWS-1;
                        if(kb_cursor_row==3) kb_cursor_col=0; // row3 => single col=0
                    }
                    if(pressed & CONT_DPAD_LEFT){
                        if(kb_cursor_col>0) kb_cursor_col--;
                    }
                    if(pressed & CONT_DPAD_RIGHT){
                        kb_cursor_col++;
                    }
                    if(kb_cursor_row==3) kb_cursor_col=0; 
                }
                // Update old button state
                old_dc_buttons = state->buttons;
            }
        }
#endif

        
        while(SDL_PollEvent(&e))
        {
            if(e.type==SDL_QUIT){
                running=0;
            }
            else if(e.type==SDL_KEYDOWN){
                if(e.key.keysym.sym== SDLK_ESCAPE){
                    running=0;
                } else if(gUIState==0){
                    // input
                    if(e.key.keysym.sym== SDLK_RETURN){
                        keyboard_select_key("[ENTER]");
                    } else if(e.key.keysym.sym== SDLK_BACKSPACE){
                        int len= strlen(gUserInput);
                        if(len>0) gUserInput[len-1]='\0';
                    } else {
                        // normal char
                        SDLKey k= e.key.keysym.sym;
                        if(k>=32 && k<127){
                            int len= strlen(gUserInput);
                            if(len<(int)sizeof(gUserInput)-1){
                                gUserInput[len]=(char)k;
                                gUserInput[len+1]='\0';
                            }
                        }
                    }
                } else {
                    // output => any key => go input
                    gUIState=0;
                }
            }
#ifndef DREAMCAST
            else if(e.type==SDL_JOYBUTTONDOWN){
                if(gUIState==0){
                    if(e.jbutton.button==0){
                        // "select"
                        const char*kk= keyboard_get_key(kb_cursor_row,kb_cursor_col);
                        keyboard_select_key(kk);
                    }
                } else {
                    gUIState=0;
                }
            }
            else if(e.type==SDL_JOYAXISMOTION){
                if(gUIState==0){
                    int axis= e.jaxis.axis;
                    int val= e.jaxis.value;
                    const int DEAD=8000;
                    int sign=0;
                    if(val< -DEAD) sign=-1;
                    else if(val>DEAD) sign=1;

                    if(axis==0){
                        if(sign!= gAxisSign[0]){
                            gAxisSign[0]= sign;
                            if(sign<0){
                                kb_cursor_col--;
                                if(kb_cursor_col<0)kb_cursor_col=0;
                            } else if(sign>0){
                                kb_cursor_col++;
                            }
                        }
                    } else if(axis==1){
                        if(sign!= gAxisSign[1]){
                            gAxisSign[1]= sign;
                            if(sign<0){
                                kb_cursor_row--;
                                if(kb_cursor_row<0)kb_cursor_row=0;
                            } else if(sign>0){
                                kb_cursor_row++;
                                if(kb_cursor_row>=KB_ROWS) kb_cursor_row=KB_ROWS-1;
                                // special case row3 => single col=0
                                if(kb_cursor_row==3) kb_cursor_col=0;
                            }
                        }
                    }
                    // clamp col=0 if row3
                    if(kb_cursor_row==3) kb_cursor_col=0;
                }
            }
#endif
        }

        main_render();
        
        SDL_Delay(16);
    }

    if(gJoystick){
        SDL_JoystickClose(gJoystick);
        gJoystick= NULL;
    }
    SDL_Quit();

    free_sampler(&gSampler);
    free_tokenizer(&gTokenizer);
    free_transformer(&gTransformer);

    return 0;
}
