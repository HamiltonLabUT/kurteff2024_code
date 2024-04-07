# Update this path locally
data <- read.csv('/path/to/git/kurteff2024_code/stats/lme/csv/temporal_vs_insular_spkr.csv')

library(emmeans)

lm <- lmer("peak_lat ~ roi + (1|subj)", data=data)
em <- emmeans(lm, pairwise ~ roi)
print(summary(em))
rnf <- ranef(lm) # random effects
print(rnf)
efs <- eff_size(em, sigma=sigma(lm), df.residual(lm)) # effect size
print(efs)