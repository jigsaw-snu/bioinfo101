install.packages("keras")
install.packages("reticulate")
devtools::install_github("mPloenzke/learnMotifs")

library(keras)
library(learnMotifs)


# Model config options
opt <- list()
opt$max_len <- 200 # sequence length
opt$log_dir <- 'log' # logging directory
opt$embedding_pos <- 'GGGGGGGG' # sub-sequence inserted into Y=1 sequences
opt$embedding_neg <- 'CCCCCCCC' # sub-sequence inserted into Y=0 sequences
opt$batch_size <- 64 # batch size for training 
opt$epochs <- 20 # training epochs
opt$n_filters <- 4 # number of convolutional filters
opt$filter_len <- 12 # convolutional filter length
opt$lambda_pos <- 3e-3 # regularization penalty for position-specific weights in filters
opt$lambda_filter <- 1e-8 # group regularization penalty treating filter as groups
opt$lambda_l1 <- 3e-3 # regularization penalty for all weights in filters
opt$lambda_offset <- 0 # regularization penalty for the offset used in the convolutional filters
opt$lambda_beta <- 3e-3 # regularization penalty for the betas (effect sizes)
opt$learning_rate <- 0.02 # stochastic gradient descent learning rate
opt$decay_rate <- opt$learning_rate / opt$epochs / 2 # decay for learning rate
opt$plot_Activations <- TRUE # boxplot of activation differences by class per filter
opt$plot_filterMotifs <- TRUE # plot sequence logos (IGMs) directly
opt$plot_empiricalMotifs <- FALSE # plot sequence logos from maximally-activated sub-sequences
opt$output_plots_every <- 5 # plotting frequency (every X epochs)
opt$downsample <- 1 # downsample training cases (provide a proportion)
opt$shuffle <- .1 # proportion of cases to shuffle (i.e. how many cases contain the wrong label)
opt$cache_Old <- TRUE # cache old results



negative.cases <- learnMotifs::EmptyBackground

# random index sampling
# rbinom : generate 'n' binomial distributed samples with sample size and generating probability
idx <- as.logical(rbinom(n=nrow(negative.cases), size=1, prob=.5))

# split positive & negative cases using sampled index
positive.cases <- negative.cases[idx, ]
negative.cases <- negative.cases[!idx, ]

# sample labeling
negative.cases$y <- 0
positive.cases$y <- 1


# embed 'GGGGGGGG' into "some" positive cases
for (irow in 1:nrow(positive.cases)) {
    
    # runif(n) : generate 'n' uniform distributed random number
    # opt$shuffle : proportion of cases to shuffle
    if (runif(1) < (1 - opt$shuffle)) {
        
        # sample : get size amount of samples from original sample list (Random Sampling)
        # opt$max_len : sequence length
        # opt$embedding_pos : 'GGGGGGGG' (working as artificial motif for positive samples)
        location <- sample(1:(opt$max_len - nchar(opt$embedding_pos)), 1) # designate random position for embedding (artificial motif)
        
        # replace part of sequence to embedding_pos ('GGGGGGGG')
        # substr("abcdef", 2, 4) ->> "bcd"
        substr(
            positive.cases[irow, 'sequence'],
            start=location,
            stop=location + nchar(opt$embedding_pos) - 1 # stop - start + 1 == len(embedding_pos)
        ) <- opt$embedding_pos
        
        positive.cases[irow, 'embeddings'] <- opt$embedding_pos
    }
}

# embed 'CCCCCCCC' into "some" negative cases
for (irow in 1:nrow(negative.cases)) {
    
    if (runif(1) < (1 - opt$shuffle)) {
        
        location <- sample(1:(opt$max_len - nchar(opt$embedding_neg)), 1)
        
        substr(
            negative.cases[irow,'sequence'],
            start=location,
            stop=location + nchar(opt$embedding_neg) - 1
        ) <- opt$embedding_neg
        
        negative.cases[irow,'embeddings'] <- opt$embedding_neg
    }
}

# rbind : same effect as 'np.vstack' do
all.cases <- rbind(positive.cases, negative.cases) # total 1000 samples
# shuffling (random sampling from same sized list)
all.cases <- all.cases[sample(1:nrow(all.cases), size=nrow(all.cases), replace=FALSE), ]


# optional downsampling (defalut : not doing)
if (opt$downsample < 1) {
    all.cases <- all.cases[sample(1:nrow(all.cases), size=opt$downsample * nrow(all.cases)), ]
}


# Split Train / Validation set
idx <- sample(1:nrow(all.cases), round(.9 * nrow(all.cases))) # 90% of total index

training_data <- all.cases[idx, c('sequence')] # training_data : 90% of total sequences -> 900 samples
training_labels <- all.cases[idx, 'y']

validation_data <- all.cases[-idx, c('sequence')] # remaining sample == validation set -> 100 samples
validation_labels <- all.cases[-idx, 'y']


# initialize output directory
learnMotifs::setup_log_dir(opt)

# save validation dataset
write.table(
    cbind(validation_data, validation_labels),
    file=file.path(opt$log_dir, 'testset.csv'),
    sep=',',
    row.names=F,
    col.names=F
)


# One-Hot Encoding
training_array <- learnMotifs::one_hot(training_data, opt$filter_len) # dimension : (900, 4, 222, 1)
validation_array <- learnMotifs::one_hot(validation_data, opt$filter_len) # dimension : (100, 4, 222, 1)
                                                                          # 100 -> number of samples
                                                                          # 4 -> 'ACGT'
                                                                          # 222 -> 200 + 2 * (filter_size - 1) => (padding) + (sequence) + (padding)
                                                                          # making same size after convolving -> add padding ; n + p - k + 1 -> p = k + 1
                                                                          # 1 -> pseudo_axis for Conv2D? ; making pseudo-channel


# Build Model
deNovo_sequence <- keras::layer_input(
    shape=c(4, opt$max_len + 2 * opt$filter_len - 2, 1), # dim(training_set) = (900, 4, 222, 1) = (900, 4, (12 - 1) + 200 + (12 - 1), 1)
                                                         # dim(validation_set) = (100, 4, 222, 1)
    name="deNovo_input"
)

deNovo_model <- deNovo_sequence %>%
    learnMotifs::layer_deNovo(
        filters=opt$n_filters, # num_filters = 4 => 4 means 'one filter per case' in here (we are testing 4 cases)
                               # 1 for positive sample discriminator, 1 for negative sample discriminator, 2 are redundant (shouldn't be used inside even if we feed them as filters)
        filter_len=opt$filter_len, # filter_length = 12 -> (4 x 12) convolution filter => 4 means 'ACGT' in here
        lambda_pos=opt$lambda_pos, # Position-wise penalty using L1 
                                   # regularization penalty for position-specific weights in filters
        lambda_filter=opt$lambda_filter, # Sparse group lasso penalty.
                                         # group regularization penalty treating filter as groups
        lambda_l1=opt$lambda_l1, # Sparse L1 penalty applied to filter weights.
                                 # regularization penalty for all weights in filters
        lambda_offset=opt$lambda_offset, # L1 penalty applied the offsets (bias) terms.
                                         # regularization penalty for the offset used in the convolutional filters
        input_shape=c(4, opt$max_len + 2 * opt$filter_len - 2, 1), # dim(training_set) = (900, 4, 222, 1)
                                                                   # dim(validation_set) = (100, 4, 222, 1)
        activation='sigmoid'
    ) %>%
    keras::layer_max_pooling_2d(
        pool_size=c(1, (opt$max_len + opt$filter_len - 1)), # (1, 200 + 12 - 1) = (1, 211)
        name='deNovo_pool'
    ) %>%
    keras::layer_flatten(name='deNovo_flatten') %>%
    keras::layer_dense(
        units=1,
        activation='sigmoid',
        kernel_regularizer=keras::regularizer_l1(l=opt$lambda_beta) # regularization factor = lambda_beta
                                                                    # regularization penalty for the betas (effect sizes)
    )

model <- keras::keras_model(
    inputs=deNovo_sequence,
    outputs=deNovo_model
)

model %>% keras::compile(
    loss="binary_crossentropy",
    optimizer=keras::optimizer_adam(
        learning_rate=opt$learning_rate,
        decay=opt$decay_rate # opt$learning_rate / opt$epochs / 2
    ),
    metrics=c("accuracy")
)


deNovo_callback <- learnMotifs::deNovo_IGM_Motifs$new(
    model=model,
    N=opt$output_plots_every, # Plotting Frequency ; 5
    log_dir=opt$log_dir,
    plot_activations=opt$plot_Activations, # boxplot of activation differences by class per filter ; TRUE
    plot_filters=opt$plot_filterMotifs, # plot sequence logos (IGMs) directly ; TRUE
    plot_crosscorrel_motifs=opt$plot_empiricalMotifs, # plot sequence logos from maximally-activated sub-sequences ; FALSE
    deNovo_data=validation_array,
    test_labels=validation_labels,
    test_seqs=validation_data,
    num_deNovo=opt$n_filters,
    filter_len=opt$filter_len
)


sequence_fit <- model %>%
    keras::fit(
        x=training_array, training_labels,
        batch_size=opt$batch_size, # 64
        epochs=opt$epochs, # 20
        validation_data=list(validation_array, validation_labels),
        shuffle=TRUE,
        callbacks=list(deNovo_callback)
    )


# save training plots
p <- plot(sequence_fit, method='ggplot2', smooth=TRUE)
ggsave(paste(opt$log_dir, "Training_loss.pdf", spe='/'), plot=p, width=15, height=7.5, units='in')
model$save(file.path(opt$log_dir, 'deNovo_model.h5'))
model$save_weights(file.path(opt$log_dir, 'deNovo_model_weights.h5'))
write.table(model$to_yaml(), file.path(opt$log_dir, 'deNovo_model.yaml'))