# Update this path locally
data <- read.csv('/path/to/git/kurteff2024_code/stats/lme/csv/onset_sustained_si.csv')

library(emmeans)

# Clip data
data_onset <- data[data$elec_type != 'no_onset',]
data_spkronly <- data_onset[data_onset$elec_type == 'spkr_only',]
data_miconly <- data_onset[data_onset$elec_type == 'mic_only',]

# Run LME (spkr only)
print("Fitting linear mixed-effects model for perception trials...")
lm <- lmer("si ~ window + roi_condensed + (1|subj)", data=data_spkronly)
em <- emmeans(lm, pairwise ~ window)
print(summary(em))
rnf <- ranef(lm) # random effects
print(rnf)
efs <- eff_size(em, sigma=sigma(lm), df.residual(lm)) # effect size
print(efs)

# Run LME (mic only)
print("Fitting linear mixed-effects model for production trials...")
lm <- lmer("si ~ window + roi_condensed + (1|subj)", data=data_miconly)
em <- emmeans(lm, pairwise ~ window)
print(summary(em))
rnf <- ranef(lm) # random effects
print(rnf)
efs <- eff_size(em, sigma=sigma(lm), df.residual(lm)) # effect size
print(efs)