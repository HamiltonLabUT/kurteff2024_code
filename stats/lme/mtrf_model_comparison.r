# Update this path locally
# Change the models if you want to run another comparison (e.g., model1_model3.csv)
data <- read.csv('/path/to/git/kurteff2024_code/stats/lme/csv/model1_model2.csv')

library(emmeans)

lm <- lmer("r ~ model + (1|subject) + (1|channel)", data=data)
em <- emmeans(lm, pairwise ~ roi)
print(summary(em))
rnf <- ranef(lm) # random effects
print(rnf)
efs <- eff_size(em, sigma=sigma(lm), df.residual(lm)) # effect size
print(efs)