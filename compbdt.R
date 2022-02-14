#s11 <- 177#473
#s10 <- 0#29
#s01 <- 2#81
#s00 <- 2#25
#r11 <- 3#22
#r10 <- 3#46
#r01 <- 0#44
#r00 <- 807#151

compbdt <- function(s11, s10, s01, s00, r11, r10, r01, r00, alpha = 0.05)
{
  
  #Number of decimal places of the results
  
  decip <- 3
  
  
  ss <- s11 + s10 + s01 + s00  
  
  rr <- r11 + r10 + r01 + r00
  
  n11 <- s11 + r11
  
  n10 <- s10 + r10
  
  n01 <- s01 + r01
  
  n00 <- s00 + r00 
  
  n <- n11 + n10 + n01 + n00
  
  if (ss == 0 | rr == 0) 
  {
    cat("\n")
    stop("There are many observed frequencies equal to zero. Introduces new values \n")
    cat("\n")
  }
  
  if (s11 < 0 | s10 < 0 | s01 < 0 | s00 < 0 | r11 < 0 | r10 < 0 | r01 < 0 | r00 < 0) 
  {
    cat("\n")
    stop("Any observed frequency can be negative. Introduces new values \n")
    cat("\n")
  }
  
  if (abs(s00 - trunc(s00)) > 0  | abs(s01 - trunc(s01)) > 0  | abs(s10 - trunc(s10)) > 0  | abs(s11 - trunc(s11)) > 0  | abs(r00 - trunc(r00)) > 0  | abs(r01 - trunc(r01)) > 0
      | abs(r10 - trunc(r10)) > 0  | abs(r11 - trunc(r11)) > 0)
  {
    cat("\n")
    stop("Observed frequencies can not have decimals. Introduces new values \n")
    cat("\n")
  }  
  
  if (n10 == 0 & n01 == 0) 
  {
    cat("\n")
    stop("There are many observed frequencies equal to zero. Introduces new values \n")
    cat("\n")
  }  
  
  if (alpha <= 0 | alpha >= 1) 
  {
    cat("\n")
    stop("Alpha error should take a value between 0 and 1. Introduces a new value \n")
    cat("\n")
  }
  
  conf <- 1 - alpha
  
  z <- qnorm(1 - alpha / 2)
  
  prev <- ss / n
  
  qrev <- 1 - prev
  
  Se1 <- (s11 + s10) / ss
  
  Se2 <- (s11 + s01) / ss
  
  Sp1 <- (r01 + r00) / rr
  
  Sp2 <- (r10 + r00) / rr
  
  Y1 <- Se1 + Sp1 - 1
  
  Y2 <- Se2 + Sp2 - 1
  
  if (Y1 <= 0 | Y2 <= 0) 
  {
    cat("\n")
    cat("Estimated Youden index of Binary Test 1 is ", round(Y1, decip), "\n")
    cat("\n") 
    cat("Estimated Youden index of Binary Test 2 is ", round(Y2, decip), "\n")
    cat("\n")
    stop("Estimated Youden index of a Binary Test must be greater than zero. There are many observed frequencies equal to zero. Introduces new values \n")
    cat("\n")
  }
  
  
  
  
  #Comparison of the accuracies (sensitivities and specificities). Se: sensitivity, Sp: specificity
  
  p11 <- s11 / n   
  p10 <- s10 / n  
  p01 <- s01 / n  
  p00 <- s00 / n
  
  q11 <- r11 / n  
  q10 <- r10 / n  
  q01 <- r01 / n  
  q00 <- r00 / n
  
  VarSe1 <- Se1 * (1- Se1) / (n * prev)
  
  VarSp1 <- Sp1 * (1- Sp1) / (n * qrev)
  
  VarSe2 <- Se2 * (1- Se2) / (n * prev)
  
  VarSp2 <- Sp2 * (1- Sp2) / (n * qrev)  
  
  Varprev <- prev * qrev / n
  
  e1 <- (s11 * s00 - s10 * s01) / ss^2
  
  e0 <- (r11 * r00 - r10 * r01) / rr^2 
  
  CovSe1Se2 <- e1 / (n * prev)
  
  CovSp1Sp2 <- e0 / (n * qrev)
  
  sigmaAc <- matrix(0, 4, 4)
  
  sigmaAc[1,1] <- VarSe1
  sigmaAc[1,2] <- 0
  sigmaAc[1,3] <- CovSe1Se2
  sigmaAc[1,4] <- 0
  
  sigmaAc[2,1] <- 0
  sigmaAc[2,2] <- VarSp1
  sigmaAc[2,3] <- 0
  sigmaAc[2,4] <- CovSp1Sp2
  
  sigmaAc[3,1] <- CovSe1Se2
  sigmaAc[3,2] <- 0
  sigmaAc[3,3] <- VarSe2
  sigmaAc[3,4] <- 0
  
  sigmaAc[4,1] <- 0
  sigmaAc[4,2] <- CovSp1Sp2
  sigmaAc[4,3] <- 0
  sigmaAc[4,4] <- VarSp2
  
  if (is.nan(det(sigmaAc)))
  {
    stop("Statistic for the global hypothesis H0: (Se1 = Se2 and Sp1 = Sp2) can not be calculeted. There are many observed frequencies equal to zero. Introduces new values \n")
  }
  
  if (det(sigmaAc) == 0)
  {
    stop("Statistic for the global hypothesis H0: (Se1 = Se2 and Sp1 = Sp2)) can not be calculeted. There are many observed frequencies equal to zero. Introduces new values \n")
  }    
  
  
  #Confidence interval for the disease prevalence      
  
  Lp <- 0.5 + ((n + z^4 / 53) / (n + z^2)) * (prev - 0.5) - (z / (n + z^2)) * sqrt(n * prev * (1 - prev) + z^2 / 4)
  
  Up <- 0.5 + ((n + z^4 / 53) / (n + z^2)) * (prev - 0.5) + (z / (n + z^2)) * sqrt(n * prev * (1 - prev) + z^2 / 4)   
  
  
  #Confidence intervals for the sensitivities and specificities      
  
  LSe1 <- 0.5 + ((ss + z^4 / 53) / (ss + z^2)) * (Se1 - 0.5) - (z / (ss + z^2)) * sqrt(ss * Se1 * (1 - Se1) + z^2 / 4)
  
  USe1 <- 0.5 + ((ss + z^4 / 53) / (ss + z^2)) * (Se1 - 0.5) + (z / (ss + z^2)) * sqrt(ss * Se1 * (1 - Se1) + z^2 / 4)
  
  LSe2 <- 0.5 + ((ss + z^4 / 53) / (ss + z^2)) * (Se2 - 0.5) - (z / (ss + z^2)) * sqrt(ss * Se2 * (1 - Se2) + z^2 / 4)
  
  USe2 <- 0.5 + ((ss + z^4 / 53) / (ss + z^2)) * (Se2 - 0.5) + (z / (ss + z^2)) * sqrt(ss * Se2 * (1 - Se2) + z^2 / 4)
  
  LSp1 <- 0.5 + ((rr + z^4 / 53) / (rr + z^2)) * (Sp1 - 0.5) - (z / (rr + z^2)) * sqrt(rr * Sp1 * (1 - Sp1) + z^2 / 4)
  
  USp1 <- 0.5 + ((rr + z^4 / 53) / (rr + z^2)) * (Sp1 - 0.5) + (z / (rr + z^2)) * sqrt(rr * Sp1 * (1 - Sp1) + z^2 / 4)
  
  LSp2 <- 0.5 + ((rr + z^4 / 53) / (rr + z^2)) * (Sp2 - 0.5) - (z / (rr + z^2)) * sqrt(rr * Sp2 * (1 - Sp2) + z^2 / 4)
  
  USp2 <- 0.5 + ((rr + z^4 / 53) / (rr + z^2)) * (Sp2 - 0.5) + (z / (rr + z^2)) * sqrt(rr * Sp2 * (1 - Sp2) + z^2 / 4)  
  
  
  #Hypothesis tests
  
  w1 <- (ss * (s10 - s01)^2) / (4 * s10 * s01 + (s11 + s00) * (s10 + s01))       
  
  w2 <- (rr * (r10 - r01)^2) / (4 * r10 * r01 + (r11 + r00) * (r10 + r01))
  
  pvalue1 <- 2 * (1 - pnorm(w1, 0, 1))
  
  pvalue2 <- 2 * (1 - pnorm(w2, 0, 1))
  
  Mcc1 <- (abs(s10 - s01) - 1)^2 / (s10 + s01)
  
  Mcc2 <- (abs(r10 - r01) - 1)^2 / (r10 + r01)
  
  pvalue3 <- 2 * (1 - pnorm(Mcc1, 0, 1))
  
  pvalue4 <- 2 * (1 - pnorm(Mcc2, 0, 1))
  
  Q1 <- ss * (s10 - s01)^2 / (4 * s10 * s01 + (s11 + s00) * (s10 + s01)) + rr * (r10 - r01)^2 / (4 * r10 * r01 + (r11 + r00) * (r10 + r01))
  
  globalpvalue1 <- (1 - pchisq(Q1, 2))     
  
  
  #Confidence intervals for the difference of the two sensitivities (specificities)        
  
  A1 <- (s10 - s01) / (ss + 2) - z * sqrt((((s10 + 1) / (ss + 2)) + ((s01 + 1) / (ss + 2)) -(((s10 + 1) / (ss + 2)) - ((s01 + 1) / (ss + 2)))^2) / (ss + 2))
  
  A2 <- (s10 - s01) / (ss + 2) + z * sqrt((((s10 + 1) / (ss + 2)) + ((s01 + 1) / (ss + 2)) -(((s10 + 1) / (ss + 2)) - ((s01 + 1) / (ss + 2)))^2) / (ss + 2))
  
  if (A1 < -1) (LSe <- -1) else (LSe <- A1)
  
  if (A2 > 1) (USe <- 1) else (USe <- A2)
  
  B1 <- (r01 - r10) / (rr + 2) - z * sqrt((((r10 + 1) / (rr + 2)) + ((r01 + 1) / (rr + 2)) -(((r01 + 1) / (rr + 2)) - ((r10 + 1) / (rr + 2)))^2) / (rr + 2)) 
  
  B2 <- (r01 - r10) / (rr + 2) + z * sqrt((((r10 + 1) / (rr + 2)) + ((r01 + 1) / (rr + 2)) -(((r01 + 1) / (rr + 2)) - ((r10 + 1) / (rr + 2)))^2) / (rr + 2))
  
  if (B1 < -1) (LSp <- -1) else (LSp <- B1)
  
  if (B2 > 1) (USp <- 1) else (USp <- B2)
  
  
  
  
  
  #Comparison of the likelihood ratios. PLR: positive likelihood ratio, NLR: negative likelihood ratio
  
  PLR1 <- Se1 / (1 - Sp1)
  
  PLR2 <- Se2 / (1 - Sp2)
  
  NLR1 <- (1 - Se1) / Sp1
  
  NLR2 <- (1 - Se2) / Sp2
  
  VarPLR1 <- VarSe1 / (1 - Sp1)^2 + (Se1^2 * VarSp1) / (1 - Sp1)^4
  
  VarPLR2 <- VarSe2 / (1 - Sp2)^2 + (Se2^2 * VarSp2) / (1 - Sp2)^4
  
  VarNLR1 <- VarSe1 / Sp1^2 + ((1 - Se1)^2 * VarSp1) / Sp1^4
  
  VarNLR2 <- VarSe2 / Sp2^2 + ((1 - Se2)^2 * VarSp2) / Sp2^4
  
  
  #Confidence intervals for the LRs      
  
  ciplr <- function(s1, s0, r1, r0)
  {
    ss1 = s1 + s0
    
    rr1 = r1 + r0
    
    nn1 = ss1 + rr1
    
    p1 = (r1 + 0.5) / (rr1 + 1)
    
    p2 = (s1 + 0.5) / (ss1 + 1)
    
    Lplr <- ((nn1 + 2) * (s1 + 0.5) * (r1 + 0.5) + (z^2 / 2) * ((ss1 + 1) * (s1 + 0.5) + (rr1 + 1) * (r1 + 0.5) - 2 * (s1 + 0.5) * (r1 + 0.5)) - z * sqrt(((nn1 + 2)^2 * (r1 + 0.5) * (s1 + 0.5) * ((s1 + r1 + 1) -
                                                                                                                                                                                                      (nn1 + 2) * p1 * p2) + (z^2 / 4) * ((ss1 + 1) * (s1 + 0.5) - (rr1 + 1) * (r1 + 0.5))^2))) /  ((r1 + 0.5) * ((nn1 + 2) * (ss1 + 1) * p1 - z^2 * ((ss1 + 1) - (r1 + 0.5))))
    
    Uplr <- ((nn1 + 2) * (r1 + 0.5) * (s1 + 0.5) + (z^2 / 2) * ((ss1 + 1) * (s1 + 0.5) + (rr1 + 1) * (r1 + 0.5) - 2 * (r1 + 0.5) * (s1 + 0.5)) + z * sqrt(((nn1 + 2)^2 * (r1 + 0.5) * (s1 + 0.5) * ((s1 + r1 + 1) -
                                                                                                                                                                                                      (nn1 + 2) * p1 * p2) + (z^2 / 4) * ((ss1 + 1) * (s1 + 0.5) - (rr1 + 1) * (r1 + 0.5))^2))) /  ((r1 + 0.5) * ((nn1 + 2) * (ss1 + 1) * p1 - z^2 * ((ss1 + 1) - (r1 + 0.5))))
    
    if (Lplr < (s1 + 0.5) / ((nn1 + 2) - (r1 + 0.5)))
    {
      Lplr <- ((s1 + 0.5) * p1 + z^2 / 2 - z * sqrt(z^2 / 4 + (s1 + 0.5) * (p1 - p2))) / ((ss1 + 1) * p1^2 + z^2)
    }
    
    if (Uplr > ((nn1 + 2) - (s1 + 0.5)) / (r1 + 0.5))
    {
      Uplr <- ((r1 + 0.5) * p2 + z^2 / 2 + z * sqrt(z^2 / 4 + (r1 + 0.5) * (p2 - p1))) / ((rr1 + 1) * p1^2)
    }
    ciPLR <- list(LPLR = Lplr, UPLR = Uplr)
  }
  
  ciPLR1 <- ciplr(s11 + s10, s01 + s00, r11 + r10, r01 + r00)
  
  ciPLR2 <- ciplr(s11 + s01, s10 + s00, r11 + r01, r10 + r00)
  
  
  
  cinlr = function(s1, s0, r1, r0)
  {
    ss1 = s1 + s0
    
    rr1 = r1 + r0
    
    nn1 = ss1 + rr1
    
    p1 = (r0 + 0.5) / (rr1 + 1) 
    
    p2 = (s0 + 0.5) / (ss1 + 1)
    
    Lnlr <- ((nn1 + 2) * (r0 + 0.5) * (s0 + 0.5) + (z^2 / 2) * ((ss1 + 1) * (s0 + 0.5) + (rr1 + 1) * (r0 + 0.5) - 2 * (r0 + 0.5) * (s0 + 0.5)) - z * sqrt(((nn1 + 2)^2 * (r0 + 0.5) * (s0 + 0.5) * ((s0 + r0 + 1) -
                                                                                                                                                                                                      (nn1 + 2) * p1 * p2) + (z^2 / 4) * ((ss1 + 1) * (s0 + 0.5) - (rr1 + 1) * (r0 + 0.5))^2))) / ((r0 + 0.5) * ((nn1 + 2) * (ss1 + 1) * p1 - z^2 * ((ss1 + 1) - (r0 + 0.5))))
    
    Unlr <- ((nn1 + 2) * (r0 + 0.5) * (s0 + 0.5) + (z^2 / 2) * ((ss1 + 1) * (s0 + 0.5) + (rr1 + 1) * (r0 + 0.5) - 2 * (r0 + 0.5) * (s0 + 0.5)) + z * sqrt(((nn1 + 2)^2 * (r0 + 0.5) * (s0 + 0.5) * ((s0 + r0 + 1) -
                                                                                                                                                                                                      (nn1 + 2) * p1 * p2) + (z^2 / 4) * ((ss1 + 1) * (s0 + 0.5) - (rr1 + 1) * (r0 + 0.5))^2))) / ((r0 + 0.5) * ((nn1 + 2) * (ss + 1) * p1 - z^2 * ((ss1 + 1) - (r0 + 0.5))))
    
    if (Lnlr < (s0 + 0.5) / ((nn1 + 2) - (r0 + 0.5)))
    {
      Lnlr <- ((s0 + 0.5) * p1 + z^2 / 2 - z * sqrt(z^2 / 4 + (s0 + 0.5) * (p1 - p2))) / ((ss1 + 1) * p1^2 + z^2)
    }
    
    if (Unlr > ((nn1 + 2) - (s0 + 0.5)) / (r0 + 0.5))
    {
      Unlr <- ((r0 + 0.5) * p2 + z^2 / 2 + z * sqrt(z^2 / 4 + (r0 + 0.5) * (p2 - p1))) / ((rr1 + 1) * p1^2)
    }
    
    ciNLR <- list(LNLR = Lnlr, UNLR = Unlr)
  }
  
  ciNLR1 <- cinlr(s11 + s10, s01 + s00, r11 + r10, r01 + r00)
  
  ciNLR2 <- cinlr(s11 + s01, s10 + s00, r11 + r01, r10 + r00)
  
  
  
  #Hypothesis tests  
  
  logposw <- log(PLR1 / PLR2)
  
  lognegw <- log(NLR1 / NLR2)
  
  logwmat <- as.matrix(c(logposw, lognegw)) 
  
  mat1 <- matrix(0, 2, 8) 
  
  mat1[1,1] <- 1 / (p10 + p11) - 1 / (p01 + p11)     
  mat1[1,2] <- 1 / (p10 + p11)   
  mat1[1,3] <- -1 / (p01 + p11)   
  mat1[1,4] <- 0
  
  mat1[1,5] <- 1 / (q01 + q11) - 1 / (q10 + q11)   
  mat1[1,6] <- -1 / (q10 + q11)   
  mat1[1,7] <- 1/(q01 + q11)   
  mat1[1,8] <- 0  
  
  mat1[2,1] <- 0
  mat1[2,2] <- -1 / (p00 + p10)
  mat1[2,3] <- 1 / (p00 + p01)
  mat1[2,4] <- 1 / (p00 + p01) - 1 / (p00 + p10)
  
  mat1[2,5] <- 0
  mat1[2,6] <- 1 / (q00 + q10)
  mat1[2,7] <- -1 / (q00 + q01)
  mat1[2,8] <- 1 / (q00 + q10) - 1 / (q00 + q01)
  
  vec1 <- vector("numeric", 8)
  
  vec1[1] <- p11
  vec1[2] <- p10
  vec1[3] <- p01
  vec1[4] <- p00
  
  vec1[5] <- q11
  vec1[6] <- q10
  vec1[7] <- q01
  vec1[8] <- q00
  
  mat2 <- matrix(0, 8, 8)
  
  mat2[1,1] <- p11
  mat2[2,2] <- p10
  mat2[3,3] <- p01
  mat2[4,4] <- p00
  
  mat2[5,5] <- q11
  mat2[6,6] <- q10
  mat2[7,7] <- q01
  mat2[8,8] <- q00
  
  sigma1 <- matrix(0, 8, 8)
  
  sigma1 <- (1 / n) * (mat2 - vec1 %*% t(vec1))
  
  sigmaLR <- matrix(0, 2, 2)
  
  sigmaLR <- mat1 %*% sigma1 %*% t(mat1)
  
  if (is.nan(det(sigmaLR)))
  {
    stop("Statistic for the global hypothesis H0: (PLR1 = PLR2 and NLR1 = NLR2) can not be calculeted. There are many observed frequencies equal to zero. Introduces new values \n")
  }
  
  if (det(sigmaLR) == 0)
  {
    stop("Statistic for the global hypothesis H0: (PLR1 = PLR2 and NLR1 = NLR2) can not be calculeted. There are many observed frequencies equal to zero. Introduces new values \n")
  }    
  
  Q2 <-  t(logwmat) %*% solve(sigmaLR) %*% logwmat
  
  globalpvalue2 <- (1 - pchisq(Q2, 2))
  
  
  z1 <- abs(logposw) / sqrt(sigmaLR[1, 1])
  
  pvalue5 <- 2 * (1 - pnorm(z1, 0, 1))
  
  
  z2 <- abs(lognegw) / sqrt(sigmaLR[2, 2])
  
  pvalue6 <- 2* (1 - pnorm(z2, 0, 1))
  
  
  #Confidence intervals for the ratio between the two positive (negative) LRs
  
  Lposw <- exp(logposw - z * sqrt(sigmaLR[1, 1]))
  
  Uposw <- exp(logposw + z * sqrt(sigmaLR[1, 1]))
  
  Lnegw <- (NLR1 / NLR2) * (1 -  z * sqrt(sigmaLR[2, 2]))
  
  Unegw <- (NLR1 / NLR2) * (1 +  z * sqrt(sigmaLR[2, 2]))
  
  
  
  
  
  #Comparison of the predictive values. PPV: positive predictive value, NPV: negative predictive value 
  
  PPV1 <- (prev * Se1) / (prev * Se1 + qrev * (1 - Sp1))
  
  PPV2 <- (prev * Se2) / (prev * Se2 + qrev * (1 - Sp2))
  
  NPV1 <- (qrev * Sp1) / (prev * (1 - Se1) + qrev * Sp1)
  
  NPV2 <- (qrev * Sp2) / (prev * (1 - Se2) + qrev * Sp2)
  
  
  #Confidence intervals for the PVs      
  
  LPPV1 <- 0.5 + ((n11 + n10 + z^4 / 53) / (n11 + n10 + z^2)) * (PPV1 - 0.5) - (z / (n11 + n10 + z^2)) * sqrt((n11 + n10) * PPV1 * (1 - PPV1) + z^2 / 4)
  
  UPPV1 <- 0.5 + ((n11 + n10 + z^4 / 53) / (n11 + n10 + z^2)) * (PPV1 - 0.5) + (z / (n11 + n10 + z^2)) * sqrt((n11 + n10) * PPV1 * (1 - PPV1) + z^2 / 4)
  
  LPPV2 <- 0.5 + ((n11 + n01 + z^4 / 53) / (n11 + n01 + z^2)) * (PPV2 - 0.5) - (z / (n11 + n01 + z^2)) * sqrt((n11 + n01) * PPV2 * (1 - PPV2) + z^2 / 4)
  
  UPPV2 <- 0.5 + ((n11 + n01 + z^4 / 53) / (n11 + n01 + z^2)) * (PPV2 - 0.5) + (z / (n11 + n01 + z^2)) * sqrt((n11 + n01) * PPV2 * (1 - PPV2) + z^2 / 4)
  
  LNPV1 <- 0.5 + ((n01 + n00 + z^4 / 53) / (n01 + n00 + z^2)) * (NPV1 - 0.5) - (z / (n01 + n00 + z^2)) * sqrt((n01 + n00) * NPV1 * (1 - NPV1) + z^2 / 4)
  
  UNPV1 <- 0.5 + ((n01 + n00 + z^4 / 53) / (n01 + n00 + z^2)) * (NPV1 - 0.5) + (z / (n01 + n00 + z^2)) * sqrt((n01 + n00) * NPV1 * (1 - NPV1) + z^2 / 4)
  
  LNPV2 <- 0.5 + ((n10 + n00 + z^4 / 53) / (n10 + n00 + z^2)) * (NPV2 - 0.5) - (z / (n10 + n00 + z^2)) * sqrt((n10 + n00) * NPV2 * (1 - NPV2) + z^2 / 4)
  
  UNPV2 <- 0.5 + ((n10 + n00 + z^4 / 53) / (n10 + n00 + z^2)) * (NPV2 - 0.5) + (z / (n10 + n00 + z^2)) * sqrt((n10 + n00) * NPV2 * (1 - NPV2) + z^2 / 4) 
  
  
  Var <- matrix(0,4,4) #Variances - covariances matrix
  
  Var[1,1] <- ((p10 + p11) * (q10 + q11)) / (n * (p10 + p11 + q10 + q11)^3)
  
  Var[1,2] <- (p01 * p10 * q11 + p11 * (q01 * (q10 + q11) + q11 * (p01 + p10 + p11 + q10 + q11))) / (n * (p01 + p11 + q01 + q11)^2 * (p10 + p11 + q10 + q11)^2)
  
  Var[1,3] <- 0 
  
  Var[1,4] <- -(p00 * (p10 + p11) * q10 + p10 * q10 * (p10 + p11 + q00 + q10) + p10 * (q00 + q10) * q11) / (n * (p00 + p10 + q00 + q10)^2 * (p10 + p11 + q10 + q11)^2)
  
  Var[2,1] <- (p01 * p10 * q11 + p11 * (q01 * (q10 + q11) + q11 * (p01 + p10 + p11 + q10 + q11))) / (n * (p01 + p11 + q01 + q11)^2 * (p10 + p11 + q10 + q11)^2)
  
  Var[2,2] <- ((p01 + p11) * (q01 + q11)) / (n * (p01 + p11 + q01 + q11)^3)
  
  Var[2,3] <- -(p00 * (p01 + p11) * q01 + p01 * q01 * (p01 + p11 + q00 + q01) + p01 * (q00 + q01) * q11) / (n * (p00 + p01 + q00 + q01)^2 * (p01 + p11 + q01 + q11)^2)
  
  Var[2,4] <- 0
  
  Var[3,1] <- 0
  
  Var[3,2] <- Var[2,3]
  
  Var[3,3] <- ((p00 + p01) * (q00 + q01)) / (n * (p00 + p01 + q00 + q01)^3)
  
  Var[3,4] <- (q00 * (p00^2 + p01 * p10 + p00 * (p01 + p10 + q00 + q01)) + p00 * (q00 + q01) * q10) / (n * (p00 + p01 + q00 + q01)^2 * (p00 + p10 + q00 + q10)^2)
  
  Var[4,1] <- Var[1,4]
  
  Var[4,2] <- 0
  
  Var[4,3] <- Var[3,4]
  
  Var[4,4] <- ((p00 + p10) * (q00 + q10)) / (n * (p00 + p10 + q00 + q10)^3)
  
  phi <- matrix(0,2,4)
  
  phi[1,1] <- 1
  phi[1,2] <- -1
  
  phi[2,3] <- 1
  phi[2,4] <- -1 
  
  PV <- as.matrix(c(PPV1, PPV2, NPV1, NPV2))
  
  dim(PV)<- c(1,4)
  
  sigmaPV <- phi %*% Var %*% t(phi)
  
  if (is.nan(det(sigmaPV)))
  {
    stop("Statistic for the global hypothesis H0: (PPV1 = PPV2 and NPV1 = NPV2) can not be calculeted. There are many observed frequencies equal to zero. Introduces new values \n")
  }
  
  if (det(sigmaPV) == 0) 
  {
    stop("Statistic for the global hypothesis H0: (PPV1 = PPV2 and NPV1 = NPV2) can not be calculeted. There are many observed frequencies equal to zero. Introduces new values \n")
  }
  
  #Hypothesis tests      
  
  Q3 <-  PV %*% t(phi) %*% solve(sigmaPV) %*% phi %*% t(PV)
  
  globalpvalue3 <- (1 - pchisq(Q3, 2)) 
  
  PPVp <- (2 * s11 + s10 + s01) / (2 * n11 + n10 + n01)
  
  NPVp <- (2 * r00 + r01 + r10) / (2 * n00 + n01 + n10)
  
  CPPVp <- (s11 * (1 - PPVp)^2 + r11 * PPVp^2) / (2 * n11 + n10 + n01)
  
  CNPVp <- (s00 * NPVp^2 + r00 * (1 - NPVp)^2) / (2 * n00 + n01 + n10)
  
  T1 <- (PPV1 - PPV2)^2 / ((PPVp * (1 - PPVp) - 2 * CPPVp) * ((1 / (n10 + n11)) + (1 / (n01 + n11))))
  
  pvalue7 <- (1 - pchisq(T1, 1))
  
  T2 <- (NPV1 - NPV2)^2 / ((NPVp * (1 - NPVp) - 2 * CNPVp) * ((1 / (n01 + n00)) + (1 / (n10 + n00))))
  
  pvalue8 <- (1 - pchisq(T2, 1))
  
  
  #Confidence intervals for the difference between the two PPVs (NPVs)
  
  C1 <- PPV1 - PPV2 - z * sqrt((PPVp * (1 - PPVp) - 2 * CPPVp) * ((1 / (n10 + n11)) + (1 / (n01 + n11))))
  
  C2 <- PPV1 - PPV2 + z * sqrt((PPVp * (1 - PPVp) - 2 * CPPVp) * ((1 / (n10 + n11)) + (1 / (n01 + n11))))
  
  if (C1 < -1) (LPPV <- -1) else (LPPV <- C1)
  
  if (C2 > 1) (UPPV <- 1) else (UPPV <- C2)
  
  D1 <- NPV1 - NPV2 - z * sqrt((NPVp * (1 - NPVp) - 2 * CNPVp) * ((1 / (n01 + n00)) + (1 / (n10 + n00))))
  
  D2 <- NPV1 - NPV2 + z * sqrt((NPVp * (1 - NPVp) - 2 * CNPVp) * ((1 / (n01 + n00)) + (1 / (n10 + n00))))
  
  if (D1 < -1) (LNPV <- -1) else (LNPV <- D1)
  
  if (D2 > 1) (UNPV <- 1) else (UNPV <- D2)
  
  
  
  
  
  
  
  #Estimation of the powers
  
  Nsample <- 10000     
  
  x0 <- 0 
  x1 <- 0   
  x2 <- 0  
  x3 <- 0
  
  j <- 1
  
  rmult <- vector("integer", 8)
  
  set.seed(1234)
  
  while (j <= Nsample)
  {         
    rmult <- rmultinom(1, n, pr = c(p11, p10, p01, p00, q11, q10, q01, q00))
    
    a11 <- rmult[1]
    a10 <- rmult[2]
    a01 <- rmult[3]
    a00 <- rmult[4]
    
    b11 <- rmult[5]
    b10 <- rmult[6]
    b01 <- rmult[7]
    b00 <- rmult[8]        
    
    aa <- a11 + a10 + a01 + a00
    
    bb <- b11 + b10 + b01 + b00 
    
    n1 <- aa + bb
    
    if (a10 + a01 == 0) validity0 <- 0 else validity0 <- 1  
    
    if (b10 + b01 == 0) validity1 <- 0 else validity1 <- 1     
    
    prev1 <- aa / n1
    
    qrev1 <- 1 - prev1  
    
    Se11 <- (a11 + a10) / aa
    
    Se21 <- (a11 + a01) / aa
    
    Sp11 <- (b01 + b00) / bb
    
    Sp21 <- (b10 + b00) / bb
    
    VarSe11 <- Se11 * (1- Se11) / (n1 * prev1)
    
    VarSp11 <- Sp11 * (1- Sp11) / (n1 * qrev1)
    
    VarSe21 <- Se21 * (1- Se21) / (n1 * prev1)
    
    VarSp21 <- Sp21 * (1- Sp21) / (n1 * qrev1)  
    
    e11 <- (a11 * a00 - a10 * a01) / aa^2
    
    e01 <- (b11 * b00 - b10 * b01) / bb^2 
    
    CovSe11Se21 <- e11 / (n1 * prev1)
    
    CovSp11Sp21 <- e01 / (n1 * qrev1)
    
    sigmaAc1 <- matrix(0, 4, 4)
    
    sigmaAc1[1,1] <- VarSe11
    sigmaAc1[1,2] <- 0
    sigmaAc1[1,3] <- CovSe11Se21
    sigmaAc1[1,4] <- 0
    
    sigmaAc1[2,1] <- 0
    sigmaAc1[2,2] <- VarSp11
    sigmaAc1[2,3] <- 0
    sigmaAc1[2,4] <- CovSp11Sp21
    
    sigmaAc1[3,1] <- CovSe11Se21
    sigmaAc1[3,2] <- 0
    sigmaAc1[3,3] <- VarSe21
    sigmaAc1[3,4] <- 0
    
    sigmaAc1[4,1] <- 0
    sigmaAc1[4,2] <- CovSp11Sp21
    sigmaAc1[4,3] <- 0
    sigmaAc1[4,4] <- VarSp21
    
    if (is.nan(det(sigmaAc1))) validity2 <-0 else validity2 <- 1          
    
    Y11 <- Se11 + Sp11 - 1
    
    Y21 <- Se21 + Sp21 - 1
    
    pLR11 <- Se11 / (1 - Sp11)
    
    pLR21 <- Se21 / (1 - Sp21)
    
    nLR11 <- (1 - Se11) / Sp11
    
    nLR21 <- (1 - Se21) / Sp21
    
    PPV11 <- (prev1 * Se11) / (prev1 * Se11 + qrev1 * (1 - Sp11))
    
    PPV21 <- (prev1 * Se21) / (prev1 * Se21 + qrev1 * (1 - Sp21))
    
    NPV11 <- (qrev1 * Sp11) / (prev1 * (1 - Se11) + qrev1 * Sp11)
    
    NPV21 <- (qrev1 * Sp21) / (prev1 * (1 - Se21) + qrev1 * Sp21)
    
    logposw1 <- log(pLR11 / pLR21)
    
    lognegw1 <- log(nLR11 / nLR21)
    
    logwmat1 <- as.matrix(c(logposw1, lognegw1)) 
    
    f11 <- a11 / n1   
    f10 <- a10 / n1  
    f01 <- a01 / n1  
    f00 <- a00 / n1
    
    h11 <- b11 / n1  
    h10 <- b10 / n1  
    h01 <- b01 / n1  
    h00 <- b00 / n1
    
    mat11 <- matrix(0, 2, 8) 
    
    mat11[1,1] <- 1 / (f10 + f11) - 1 / (f01 + f11)     
    mat11[1,2] <- 1 / (f10 + f11)   
    mat11[1,3] <- -1 / (f01 + f11)   
    mat11[1,4] <- 0  
    
    mat11[1,5] <- 1 / (h01 + h11) - 1 / (h10 + h11)   
    mat11[1,6] <- -1 / (h10 + h11)   
    mat11[1,7] <- 1/(h01 + h11)   
    mat11[1,8] <- 0  
    
    mat11[2,1] <- 0
    mat11[2,2] <- -1 / (f00 + f10)
    mat11[2,3] <- 1 / (f00 + f01)
    mat11[2,4] <- 1 / (f00 + f01) - 1 / (f00 + f10)
    
    mat11[2,5] <- 0
    mat11[2,6] <- 1 / (h00 + h10)
    mat11[2,7] <- -1 / (h00 + h01)
    mat11[2,8] <- 1 / (h00 + h10) - 1 / (h00 + h01)
    
    vec11 <- vector("numeric", 8)
    
    vec1[1] <- f11
    vec1[2] <- f10
    vec1[3] <- f01
    vec1[4] <- f00
    
    vec1[5] <- h11
    vec1[6] <- h10
    vec1[7] <- h01
    vec1[8] <- h00
    
    mat21 <- matrix(0, 8, 8)
    
    mat21[1,1] <- f11
    mat21[2,2] <- f10
    mat21[3,3] <- f01
    mat21[4,4] <- f00
    
    mat21[5,5] <- h11
    mat21[6,6] <- h10
    mat21[7,7] <- h01
    mat21[8,8] <- h00
    
    sigma11 <- matrix(0, 8, 8)
    
    sigma11 <- (1 / n1) * (mat21 - vec11 %*% t(vec11))
    
    sigmaLR1 <- matrix(0, 2, 2)
    
    sigmaLR1 <- mat11 %*% sigma11 %*% t(mat11)        
    
    Var1 <- matrix(0, 4, 4)
    
    Var1[1,1] <- ((f10 + f11) * (h10 + h11)) / (n1 * (f10 + f11 + h10 + h11)^3)
    
    Var1[1,2] <- (f01 * f10 * h11 + f11 * (h01 * (h10 + h11) + h11 * (f01 + f10 + f11 + h10 + h11))) / (n1 * (f01 + f11 + h01 + h11)^2 * (f10 + f11 + h10 + h11)^2)
    
    Var1[1,3] <- 0 
    
    Var1[1,4] <- -(f00 * (f10 + f11) * h10 + f10 * h10 * (f10 + f11 + h00 + h10) + f10 * (h00 + h10) * h11) / (n1 * (f00 + f10 + h00 + h10)^2 * (f10 + f11 + h10 + h11)^2)
    
    Var1[2,1] <- (f01 * f10 * h11 + f11 * (h01 * (h10 + h11) + h11 * (f01 + f10 + f11 + h10 + h11))) / (n1 * (f01 + f11 + h01 + h11)^2 * (f10 + f11 + h10 + h11)^2)
    
    Var1[2,2] <- ((f01 + f11) * (h01 + h11)) / (n1 * (f01 + f11 + h01 + h11)^3)
    
    Var1[2,3] <- -(f00 * (f01 + f11) * h01 + f01 * h01 * (f01 + f11 + h00 + h01) + f01 * (h00 + h01) * h11) / (n1 * (f00 + f01 + h00 + h01)^2 * (f01 + f11 + h01 + h11)^2)
    
    Var1[2,4] <- 0
    
    Var1[3,1] <- 0
    
    Var1[3,2] <- Var[2,3]
    
    Var1[3,3] <- ((f00 + f01) * (h00 + h01)) / (n1 * (f00 + f01 + h00 + h01)^3)
    
    Var1[3,4] <- (f00 * (f00^2 + f01 * f10 + f00 * (f01 + f10 + h00 + h01)) + f00 * (h00 + h01) * h10) / (n1 * (f00 + f01 + h00 + h01)^2 * (f00 + f10 + h00 + h10)^2)
    
    Var1[4,1] <- Var[1,4]
    
    Var1[4,2] <- 0
    
    Var1[4,3] <- Var[3,4]
    
    Var1[4,4] <- ((f00 + f10) * (h00 + h10)) / (n1 * (f00 + f10 + h00 + h10)^3)
    
    phi1 <- matrix(0,2,4)
    
    phi1[1,1] <- 1
    phi1[1,2] <- -1
    
    phi1[2,3] <- 1
    phi1[2,4] <- -1 
    
    PV1 <- as.matrix(c(PPV11, PPV21, NPV11, NPV21))
    
    dim(PV1)<- c(1,4)
    
    det <- det(phi1 %*% Var1 %*% t(phi1))
    
    sigmaPV1 <- phi1 %*% Var1 %*% t(phi1)
    
    if (is.nan(det(sigmaLR1))) validity3 <-0 else validity3 <- 1 
    
    if (is.nan(det(sigmaPV1))) validity4 <-0 else validity4 <- 1 
    
    validity <- validity0 * validity1 * validity2 * validity3 * validity4   
    
    if (validity > 0 & Y11 > 0 & Y21 > 0 & det(sigmaAc1) > 0 & det(sigmaLR1) > 0 & det(sigmaPV1) > 0)
    {    
      
      w11 <- (aa * (a10 - a01)^2) / (4 * a10 * a01 + (a11 + a00) * (a10 + a01))       
      
      w21 <- (bb * (b10 - b01)^2) / (4 * b10 * b01 + (b11 + b00) * (b10 + b01))
      
      pvalue11 <- 2 * (1 - pnorm(w11, 0, 1))
      
      pvalue21 <- 2 * (1 - pnorm(w21, 0, 1))
      
      if(min(pvalue11, pvalue21) <= alpha / 2) (x0 <- x0 + 1)
      
      Q11 <- aa * (a10 - a01)^2 / (4 * a10 * a01 + (a11 + a00) * (a10 + a01)) + bb * (b10 - b01)^2 / (4 * b10 * b01 + (b11 + b00) * (b10 + b01)) 
      
      globalpvalue11 <- (1 - pchisq(Q11, 2))
      
      if(globalpvalue11 <= alpha) (x1 <- x1 + 1)  
      
      Q21 <-  t(logwmat1) %*% solve(sigmaLR1) %*% logwmat1
      
      globalpvalue21 <- (1 - pchisq(Q21, 2))
      
      if(globalpvalue21 <= alpha) (x2 <- x2 + 1) else 0
      
      Q31 <-  PV1 %*% t(phi1) %*% solve(sigmaPV1) %*% phi1 %*% t(PV1)
      
      globalpvalue31 <- (1 - pchisq(Q31, 2))
      
      if(globalpvalue31 <= alpha) (x3 <- x3 + 1)
      
      j <- j + 1
    }                               
  } 
  
  y0 <- x0 / Nsample  
  
  y1 <- x1 / Nsample
  
  y2 <- x2 / Nsample
  
  y3 <- x3 / Nsample     
  
  
  
  
  # Results of the comparison of the accuracies
  sink("Results_Comparison_Accuracies.txt", split = TRUE)
  cat("\n")  
  cat("          PREVALENCE OF THE DISEASE \n")
  cat("\n")       
  cat("Estimated prevalence of the disease is ",round(100 * prev, decip),"% and its standard error is", round(sqrt(Varprev), decip), "\n")       
  cat("\n")
  cat(100 * conf,"% confidence interval for the prevalence of the disease is (", round(100 * max(0, Lp), decip),"% ; ", round(100 * min(Up, 1), decip),"%) \n")
  cat("\n")
  cat("\n") 
  cat("          COMPARISON OF THE ACCURACIES (SENSITIVITIES AND SPECIFICITIES) \n")
  cat("\n")       
  cat("Estimated sensitivity of Test 1 is ",round(100 * Se1, decip),"% and its standard error is", round(sqrt(VarSe1), decip), "\n")       
  cat("\n")
  cat(100 * conf,"% confidence interval for the sensitivity of Test 1 is (", round(100 * max(0, LSe1), decip),"% ; ", round(100 * min(USe1, 1), decip),"%) \n")
  cat("\n")
  cat("Estimated sensitivity of Test 2 is ",round(100 * Se2, decip),"% and its standard error is", round(sqrt(VarSe2), decip), "\n")       
  cat("\n")
  cat(100 * conf,"% confidence interval for the sensitivity of Test 1 is (", round(100 * max(0, LSe2), decip),"% ; ", round(100 * min(USe2, 1), decip),"%) \n")
  cat("\n")        
  cat("Estimated specificity of Test 1 is ",round(100 * Sp1, decip),"% and its standard error is", round(sqrt(VarSp1), decip), "\n")    
  cat("\n")
  cat(100 * conf,"% confidence interval for the specificity of Test 1 is (", round(100 * max(0, LSp1), decip),"% ; ", round(100 * min(USp1, 1), decip),"%) \n")
  cat("\n") 
  cat("Estimated specificity of Test 2 is ",round(100 * Sp2, decip),"% and its standard error is", round(sqrt(VarSp2), decip), "\n")    
  cat("\n")
  cat(100 * conf,"% confidence interval for the specificity of Test 1 is (", round(100 * max(0, LSp2), decip),"% ; ", round(100 * min(USp2, 1), decip),"%) \n")
  cat("\n")
  cat("\n")
  
  
  if (prev <= 0.10  & n <= 100)
  {     
    if (min(pvalue1, pvalue2) > alpha / 2 )
    {
      cat("Wald test statistic for H0: Se1 = Se2 is ", round(w1, decip)," and the two-sided p-value is ", round(pvalue1, decip)," \n")
      cat("\n")
      cat("Wald test statistic for H0: Sp1 = Sp2 is ", round(w2, decip)," and the two-sided p-value is ", round(pvalue2, decip)," \n")
      cat("\n")
      cat("  Applying the Holm method (to an alpha error of ", 100 * alpha," %), we do not reject the hypothesis H0: Se1 = Se2 and the hypothesis H0: Sp1 = Sp2 \n")
      cat("\n")
      if (abs(Se1 -Se2) > 0 & abs(Sp1 - Sp2) > 0)
      {
        cat("  Estimated probability of committing a type II error (to an alpha error of", 100 * alpha,"%) is", round(100 * y0, decip),"%  \n")
      }
      cat("\n")
    }
    
    if (min(pvalue1, pvalue2) <= alpha / 2 & max(pvalue1, pvalue2) <= alpha)
    {
      cat("Wald test statistic for H0: Se1 = Se2 is ", w1," and the two-sided p-value is ", round(pvalue1, decip)," \n")
      cat("\n")
      cat("Wald test statistic for H0: Sp1 = Sp2 is ", w2," and the two-sided p-value is ", round(pvalue2, decip)," \n")
      cat("\n")
      cat("  Applying the Holm method (to an alpha error of ", 100 * alpha," %), we reject the hypothesis H0: Se1 = Se2 and the hypothesis H0: Sp1 = Sp2 \n")
      cat("\n")
      if (abs(Se1 -Se2) > 0 & abs(Sp1 - Sp2) > 0)
      {
        cat("  Estimated power (to an alpha error of", 100 * alpha,"%) is", round(100 * y0, decip),"%  \n")
      }  
      cat("\n")
      
      if(Se1 > Se2)
      {
        cat("   Sensitivity of Test 1 is significantly greater than sensitivity of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference Se1 - Se2 is (", round(100 * LSe, decip),"% ; ", round(100 * USe, decip),"%) \n")          
        cat("\n")
      }
      else
      {
        cat("   Sensitivity of Test 2 is significantly greater than sensitivity of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference Se2 - Se1 is (", round(100 * (-USe), decip),"% ; ",round(100 * (-LSe), decip),"%) \n")          
        cat("\n")     
      }
      
      if(Sp1 > Sp2)
      {
        cat("   Specificity of Test 1 is significantly greater than specificity of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference Sp1 - Sp2 is (", round(100 * LSp, decip),"% ; ", round(100 * USp, decip),"%) \n")          
        cat("\n")
      }
      else
      {
        cat("   Specificity of Test 2 is significantly greater than specificity of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference Sp2 - Sp1 is (", round(100 * (-USp), decip),"% ; ", round(100 * (-LSp), decip),"%) \n")          
        cat("\n")     
      }
    }
    
    if (pvalue1 <= alpha / 2 & pvalue2 > alpha)
    {
      cat("Wald test statistic for H0: Se1 = Se2 is ", w1," and the two-sided p-value is ", pvalue1," \n")
      cat("\n")
      cat("Wald test statistic for H0: Sp1 = Sp2 is ", w2," and the two-sided p-value is ", pvalue2," \n")
      cat("\n")
      cat("  Applying the Holm method (to an alpha error of ", 100 * alpha," %), we reject the hypothesis H0: Se1 = Se2 and we do not reject the hypothesis H0: Sp1 = Sp2 \n")
      cat("\n")
      if (abs(Se1 -Se2) > 0 & abs(Sp1 - Sp2) > 0)
      {
        cat("  Estimated power (to an alpha error of", 100 * alpha,"%) is", round(100 * y0, decip),"%  \n")
      }  
      cat("\n")
      
      if(Se1 > Se2)
      {
        cat("   Sensitivity of Test 1 is significantly greater than sensitivity of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference Se1 - Se2 is (", round(100 * LSe, decip),"% ; ", round(100 * USe, decip),"%) \n")          
        cat("\n")
      }
      else
      {
        cat("   Sensitivity of Test 2 is significantly greater than sensitivity of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference Se2 - Se1 is (", round(100 * (-USe), decip),"% ; ", round(100 * (-LSe), decip),"%) \n")          
        cat("\n")     
      }
    }    
    
    if (pvalue1 > alpha & pvalue2 <= alpha / 2)
    {
      cat("Wald test statistic for H0: Se1 = Se2 is ", round(w1, decip)," and the two-sided p-value is ", round(pvalue1, decip)," \n")
      cat("\n")
      cat("Wald test statistic for H0: Sp1 = Sp2 is ", round(w2, decip)," and the two-sided p-value is ", round(pvalue2, decip)," \n")
      cat("\n")
      cat("   Applying the Holm method (to an alpha error of ", 100 * alpha," %), we do not reject the hypothesis H0: Se1 = Se2 and we reject the hypothesis H0: Sp1 = Sp2 \n")
      cat("\n")
      if (abs(Se1 -Se2) > 0 & abs(Sp1 - Sp2) > 0)
      {
        cat("  Estimated power (to an alpha error of", 100 * alpha,"%) is", round(100 * y0, decip),"%  \n")
      }  
      cat("\n")
      
      if(Sp1 > Sp2)
      {
        cat("   Specificity of Test 1 is significantly greater than specificity of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference Sp1 - Sp2 is (", round(100 * LSp, decip),"% ; ", round(100 * USp, decip),"%) \n")          
        cat("\n")
      }
      else
      {
        cat("   Specificity of Test 2 is significantly greater than specificity of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference Sp2 - Sp1 is (", round(100 * (-USp), decip),"% ; ", round(100 * (-LSp), decip),"%) \n")          
        cat("\n")     
      }
    }  
  }
  
  else
  {
    if (globalpvalue1 > alpha)
    {            
      cat("Wald test statistic for the global hypothesis test H0: (Se1 = Se2 and Sp1 = Sp2) is ", round(Q1, decip)," \n")
      cat("\n")
      cat("  Global p-value is ", round(globalpvalue1, decip)," \n")
      cat("\n")
      cat("  Applying the Wald test (to an alpha error of", 100 * alpha,"%), we do not reject the global hypothesis H0: (Se1 = Se2 and Sp1 = Sp2) \n")
      cat("\n")
      if (abs(Se1 -Se2) > 0 & abs(Sp1 - Sp2) > 0)
      {
        cat("  Estimated probability of committing a type II error (to an alpha error of", 100 * alpha,"%) is", round(100 * y1, decip),"%  \n")
      }
      cat("\n")
      
    }
    else
    {
      cat("Wald test statistic for the global hypothesis test H0: (Se1 = Se2 and Sp1 = Sp2) is ", round(Q1, decip)," \n")
      cat("\n")
      cat("  Global p-value is ", round(globalpvalue1, decip)," \n")
      cat("\n")
      cat("  Applying the global Wald test (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: (Se1 = Se2 and Sp1 = Sp2) \n")
      cat("\n")
      if (abs(Se1 -Se2) > 0 & abs(Sp1 - Sp2) > 0)
      {
        cat("  Estimated power (to an alpha error of", 100 * alpha,"%) is", round(100 * y1, decip),"%  \n")
      }          
      cat("\n")            
      cat("  Investigation of the causes of significance: \n")
      cat("\n")
      
      if (n <= 100 | n>= 1000) 
      {
        cat("   Wald test statistic for H0: Se1 = Se2 is ", round(w1, decip)," and the two-sided p-value is ", round(pvalue1, decip)," \n")
        cat("\n")
        cat("   Wald test statistic for H0: Sp1 = Sp2 is ", round(w2, decip)," and the two-sided p-value is ", round(pvalue2, decip)," \n")
        cat("\n")
        
        if (min(pvalue1, pvalue2) <= alpha / 2 & max(pvalue1, pvalue2) <= alpha)
        {
          cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: Se1 = Se2 and the hypothesis H0: Sp1 = Sp2 \n")
          cat("\n")
          
          if(Se1 > Se2)
          {
            cat("   Sensitivity of Test 1 is significantly greater than sensitivity of Test 2 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Se1 - Se2 is (", round(100 * LSe, decip),"% ; ", round(100 * USe, decip),"%) \n")          
            cat("\n")
          }
          else
          {
            cat("   Sensitivity of Test 2 is significantly greater than sensitivity of Test 1 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Se2 - Se1 is (", round(100 * (-USe), decip),"% ; ", round(100 * (-LSe), decip),"%) \n")          
            cat("\n")     
          }
          
          if(Sp1 > Sp2)
          {
            cat("   Specificity of Test 1 is significantly greater than specificity of Test 2 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Sp1 - Sp2 is (", round(100 * LSp, decip),"% ; ", round(100 * USp, decip),"%) \n")          
            cat("\n")
          }
          else
          {
            cat("   Specificity of Test 2 is significantly greater than specificity of Test 1 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Sp2 - Sp1 is (", round(100 * (-USp), decip)," ; ", round(100 * (-LSp), decip),"%) \n")          
            cat("\n")     
          }
        }
        
        if (pvalue1 <= alpha / 2 & pvalue2 > alpha)
        {
          cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: Se1 = Se2 and we do not reject the hypothesis H0: Sp1 = Sp2 \n")
          cat("\n")
          
          if(Se1 > Se2)
          {
            cat("   Sensitivity of Test 1 is significantly greater than Sensitivity of Test 2 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Se1 - Se2 is (", round(100 * LSe, decip),"% ; ", round(100 * USe, decip),"%) \n")          
            cat("\n")
          }
          else
          {
            cat("   Sensitivity of Test 2 is significantly greater than sensitivity of Test 1 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Se2 - Se1 is (", round(100 * (-USe), decip),"% ; ", round(100 * (-LSe), decip),"%) \n")          
            cat("\n")     
          }
        }    
        
        if (pvalue1 > alpha & pvalue2 <= alpha / 2)
        {
          cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we do not reject the hypothesis H0: Se1 = Se2 and we reject the hypothesis H0: Sp1 = Sp2 \n")
          cat("\n")
          
          if(Sp1 > Sp2)
          {
            cat("   Specificity of Test 1 is significantly greater than specificity of Test 2 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Sp1 - Sp2 is (", round(100 * LSp, decip),"% ; ", round(100 * USp, decip),"%) \n")          
            cat("\n")
          }
          else
          {
            cat("   Specificity of Test 2 is significantly greater than specificity of Test 1 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Sp2 - Sp1 is (", round(100 * (-USp), decip),"% ; ", round(100 * (-LSp), decip),"%) \n")          
            cat("\n")     
          }
        }                 
      }
      
      else 
      {
        cat("   McNemar test statistic (with cc) for H0: Se1 = Se2 is ", round(Mcc1, decip)," and the two-sided p-value is ", round(pvalue3, decip)," \n")
        cat("\n")
        cat("   McNemar test statistic (with cc) for H0: Sp1 = Sp2 is ", round(Mcc2, decip)," and the two-sided p-value is ", round(pvalue4, decip)," \n")
        cat("\n")
        
        if (min(pvalue3, pvalue4) <= alpha / 2 & max(pvalue3, pvalue4) <= alpha)
        {
          cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: Se1 = Se2 and the hypothesis H0: Sp1 = Sp2 \n")
          cat("\n")
          
          if(Se1 > Se2)
          {
            cat("   Sensitivity of Test 1 is significantly greater than sensitivity of Test 2 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Se1 - Se2 is (", round(100 * LSe, decip),"% ; ", round(100 * USe, decip),"%) \n")          
            cat("\n")
          }
          else
          {
            cat("   Sensitivity of Test 2 is significantly greater than sensitivity of Test 1 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Se2 - Se1 is (", round(100 * (-USe), decip),"% ; ", round(100 * (-LSe), decip),"%) \n")          
            cat("\n")     
          }
          
          if(Sp1 > Sp2)
          {
            cat("   Specificity of Test 1 is significantly greater than specificity of Test 2 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Sp1 - Sp2 is (", round(100 * LSp, decip),"% ; ", round(100 * USp, decip),"%) \n")          
            cat("\n")
          }
          else
          {
            cat("   Specificity of Test 2 is significantly greater than specificity of Test 1 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Sp2 - Sp1 is (", round(100 * (-USp), decip)," ; ", round(100 * (-LSp), decip),"%) \n")          
            cat("\n")     
          }
        }
        
        if (pvalue3 <= alpha / 2 & pvalue4 > alpha)
        {
          cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: Se1 = Se2 and we do not reject the hypothesis H0: Sp1 = Sp2 \n")
          cat("\n")
          
          if(Se1 > Se2)
          {
            cat("   Sensitivity of Test 1 is significantly greater than sensitivity of Test 2 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Se1 - Se2 is (", round(100 * LSe, decip),"% ; ", round(100 * USe, decip),"%) \n")          
            cat("\n")
          }
          else
          {
            cat("   Sensitivity of Test 2 is significantly greater than sensitivity of Test 1 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Se2 - Se1 is (", round(100 * (-USe), decip),"% ; ", round(100 * (-LSe), decip),"%) \n")          
            cat("\n")     
          }
        }    
        
        if (pvalue3 > alpha & pvalue4 <= alpha / 2)
        {
          cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we do not reject the hypothesis H0: Se1 = Se2 and we reject the hypothesis H0: Sp1 = Sp2 \n")
          cat("\n")
          
          if(Sp1 > Sp2)
          {
            cat("   Specificity of Test 1 is significantly greater than specificity of Test 2 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Sp1 - Sp2 is (", round(100 * LSp, dcip),"% ; ", round(100 * USp, decip),"%) \n")          
            cat("\n")
          }
          else
          {
            cat("   Specificity of Test 2 is significantly greater than specificity of Test 1 \n")
            cat("\n")                   
            cat("   ",100 * conf,"% confidence interval for the difference Sp2 - Sp1 is (", round(100 * (-USp), decip),"% ; ", round(100 * (-LSp), decip),"%) \n")          
            cat("\n")     
          }
        }                 
      } 
    }
  }         
  
  sink()     
  
  
  
  
  #Results of the comparison of the likelihood ratios
  sink("Results_Comparison_LRs.txt", split = TRUE)
  cat("\n") 
  cat("\n") 
  cat("          COMPARISON OF THE LIKELIHOOD RATIOS \n")
  cat("\n")       
  cat("Estimated positive LR of Test 1 is ", round(PLR1, decip)," and its standard error is", round(sqrt(VarPLR1), decip), "\n")       
  cat("\n")
  cat(100 * conf,"% confidence interval for the positive LR of Test 1 is (", round(ciPLR1$LPLR, decip)," ; ", round(ciPLR1$UPLR, decip),") \n")
  cat("\n") 
  cat("Estimated positive LR of Test 2 is ", round(PLR2, decip)," and its standard error is", round(sqrt(VarPLR2), decip), "\n")       
  cat("\n")
  cat(100 * conf,"% confidence interval for the positive LR of Test 1 is (", round(ciPLR2$LPLR, decip)," ; ", round(ciPLR2$UPLR, decip),") \n")
  cat("\n")  
  cat("Estimated negative LR of Test 1 is ", round(NLR1, decip)," and its standard error is", round(sqrt(VarNLR1), decip), "\n")   
  cat("\n") 
  cat(100 * conf,"% confidence interval for the negative LR of Test 1 is (", round(ciNLR1$LNLR, decip)," ; ", round(ciNLR1$UNLR, decip),") \n")
  cat("\n")
  cat("Estimated negative LR of Test 2 is ", round(NLR2, decip)," and its standard error is", round(sqrt(VarNLR2), decip), "\n")                
  cat("\n")
  cat(100 * conf,"% confidence interval for the negative LR of Test 2 is (", round(ciNLR2$LNLR, decip)," ; ", round(ciNLR2$UNLR, decip),") \n")
  cat("\n")
  cat("\n")
  
  if (globalpvalue2 > alpha)
  {
    cat("Test statistic for the global hypothesis test H0: (PLR1 = PLR2 and NLR1 = NLR2) is ", round(Q2, decip)," \n")
    cat("\n")
    cat("  Global p-value is ", round(globalpvalue2, decip)," \n")
    cat("\n")
    cat("  Applying the global hypothesis test (to an alpha error of", 100 * alpha,"%), we do not reject the hypothesis H0: (PLR1 = PLR2 and NLR1 = NLR2) \n")
    cat("\n")
    if (abs(PLR1 - PLR2) > 0 & abs(NLR1 - NLR2) > 0)
    {
      cat("  Estimated probability of committing a type II error (to an alpha error of", 100 * alpha,"%) is", round(100 * y2, decip),"%  \n")
    }
    cat("\n")      
  }
  else
  {
    cat("Test statistic for the global hypothesis test H0: (PLR1 = PLR2 and NLR1 = NLR2) is ",round(Q2, decip)," \n")
    cat("\n")
    cat("  Global p-value is ", round(globalpvalue2, decip)," \n")
    cat("\n")
    cat("  Applying the global hypothesis test (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: (PLR1 = PLR2 and NLR1 = NLR2) \n")
    cat("\n")
    if (abs(PLR1 - PLR2) > 0 & abs(NLR1 - NLR2) > 0)
    {
      cat("  Estimated power (to an alpha error of", 100 * alpha,"%) is", round(100 * y2, decip),"%  \n")
    }          
    cat("\n") 
    cat("  Investigation of the causes of significance: \n")
    cat("\n")
    cat("   Test statistic for H0: PLR1 = PLR2 is ", round(z1, decip)," and the two-sided p-value is ", round(pvalue5, decip)," \n")
    cat("\n")
    cat("   Test statistic for H0: NLR1 = NLR2 is ", round(z2, decip)," and the two-sided p-value is ", round(pvalue6, decip)," \n")
    cat("\n")  
    
    if (min(pvalue5, pvalue6) <= alpha / 2 & max(pvalue5, pvalue6) <= alpha)
    {
      cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: PLR1 = PLR2 and the hypothesis H0: NLR1 = NLR2 \n")
      cat("\n")
      
      if(PLR1 > PLR2)
      {
        cat("   Positive likelihood ratio of Test 1 is significantly greater than positive likelihood ratio of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the ratio PLR1 / PLR2 is (", round(Lposw, decip)," ; ", round(Uposw, decip),") \n")          
        cat("\n")
      }
      else
      {
        cat("   Positive likelihood ratio of Test 2 is significantly greater than positive likelihood ratio of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the ratio PLR2 / PLR1 is (", round(1 / Uposw, decip)," ; ", round(1 / Lposw, decip),") \n")          
        cat("\n")     
      }
      
      if(NLR1 > NLR2)
      {
        cat("   Negative likelihood ratio of Test 1 is significantly greater than negative likelihood ratio of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the ratio NLR1 / NLR2 is (", round(Lnegw, decip)," ; ", round(Unegw, decip),") \n")          
        cat("\n")
      }
      else
      {
        cat("   Negative likelihood ratio of Test 2 is significantly greater than negative likelihood ratio of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the ratio NLR2 / NLR1 is (", round(Lnegw * (nLR2 / nLR1)^2, decip)," ; ", round(Unegw * (nLR2 / nLR1)^2, decip),") \n")          
        cat("\n")     
      }
    }
    
    if (pvalue5 <= alpha / 2 & pvalue6 > alpha)
    {
      cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: PLR1 = PLR2 and we do not reject the hypothesis H0: NLR1 = NLR2 \n")
      cat("\n")
      
      if(PLR1 > PLR2)
      {
        cat("   Positive likelihood ratio of Test 1 is significantly greater than positive likelihood ratio of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the ratio PLR1 / PLR2 is (", round(Lposw, decip)," ; ", round(Uposw, decip),") \n")          
        cat("\n")
      }
      else
      {
        cat("   Positive likelihood ratio of Test 2 is significantly greater than positive likelihood ratio of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the ratio PLR2 / PLR1 is (", round(1 / Uposw, decip)," ; ", round(1 / Lposw, decip),") \n")          
        cat("\n")     
      }
    }
    
    if (pvalue5 > alpha & pvalue6 <= alpha / 2)
    {
      cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we do not reject the hypothesis H0: PLR1 = PLR2 and we reject the hypothesis H0: NLR1 = NLR2 \n")
      cat("\n")
      
      if(NLR1 > NLR2)
      {
        cat("   Negative likelihood ratio of Test 1 is significantly greater than negative likelihood ratio of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the ratio NLR1 / NLR2 is (", round(Lnegw, decip)," ; ", round(Unegw, decip),") \n")          
        cat("\n")
      }
      else
      {
        cat("   Negative likelihood ratio of Test 2 is significantly greater than negative likelihood ratio of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the ratio NLR2 / NLR1 is (", round(Lnegw * (nLR2 / nLR1)^2, decip)," ; ", round(Unegw * (nLR2 / nLR1)^2, decip),") \n")          
        cat("\n")     
      }
    }              
  }              
  sink()
  
  
  
  
  #Results of the comparison of the predictive values
  sink("Results_Comparison_PVs.txt", split = TRUE)
  cat("\n") 
  cat("\n") 
  cat("          COMPARISON OF THE PREDICTIVE VALUES \n")
  cat("\n")       
  cat("Estimated positive PV of Test 1 is ", round(100 * PPV1, decip),"% and its standard error is", round(sqrt(Var[1, 1]), decip), "\n")       
  cat("\n")
  cat(100 * conf,"% confidence interval for the positive PV of Test 1 is (", round(100 * max(0, LPPV1), decip),"% ; ", round(100 * min(UPPV1, 1), decip),"%) \n")
  cat("\n")  
  cat("Estimated positive PV of Test 2 is ", round(100 * PPV2, decip),"% and its standard error is", round(sqrt(Var[2, 2]), decip), "\n")    
  cat("\n")
  cat(100 * conf,"% confidence interval for the positive PV of Test 2 is (", round(100 * max(0, LPPV2), decip),"% ; ", round(100 * min(UPPV2, 1), decip),"%) \n")
  cat("\n") 
  cat("Estimated negative PV of Test 1 is ",round(100 * NPV1, decip),"% and its standard error is", round(sqrt(Var[3, 3]), decip), "\n")    
  cat("\n")
  cat(100 * conf,"% confidence interval for the negative PV of Test 1 is (", round(100 * max(0, LNPV1), decip),"% ; ", round(100 * min(UNPV1, 1), decip),"%) \n")
  cat("\n") 
  cat("Estimated negative PV of Test 2 is ",round(100 * NPV2, decip),"% and its standard error is", round(sqrt(Var[4, 4]), decip), "\n")              
  cat("\n")
  cat(100 * conf,"% confidence interval for the negative PV of Test 2 is (", round(100 * max(0, LNPV2), decip),"% ; ", round(100 * min(UNPV2, 1), decip),"%) \n")
  cat("\n")
  cat("\n")
  
  if (globalpvalue3 > alpha)
  {
    cat("Wald test statistic for the global hypothesis test H0: (PPV1 = PPV2 and NPV1 = NPV2) is ", round(Q3, decip)," \n")
    cat("\n")
    cat("  Global p-value is ", round(globalpvalue3, decip)," \n")
    cat("\n")
    cat("  Applying the global hypothesis test (to an alpha error of", 100 * alpha,"%), we do not reject the hypothesis H0: (PPV1 = PPV2 and NPV1 = NPV2) \n")
    cat("\n")
    if (abs(PPV1 - PPV2) > 0 & abs(NPV1 - NPV2) > 0)
    {
      cat("  Estimated probability of committing a type II error (to an alpha error of", 100 * alpha,"%) is", round(100 * y3, decip),"%  \n")
    }          
    cat("\n")      
  }
  else
  {
    cat("Wald test statistic for the global hypothesis test H0: (PPV1 = PPV2 and NPV1 = NPV2) is ", round(Q3, decip)," \n")
    cat("\n")
    cat("  Global p-value is ", round(globalpvalue3, decip)," \n")
    cat("\n")
    cat("  Applying the global hypothesis test (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: (PPV1 = PPV2 and NPV1 = NPV2) \n")
    cat("\n")
    if (abs(PPV1 - PPV2) > 0 & abs(NPV1 - NPV2) > 0)
    {
      cat("  Estimated power (to an alpha error of", 100 * alpha,"%) is", round(100 * y3, decip),"%  \n")
    } 
    cat("\n") 
    cat("  Investigation of the causes of significance: \n")
    cat("\n")
    cat("   Weighted generalized score statistic for H0: PPV1 = PPV2 is ", round(T1, decip)," and the two-sided p-value is ", round(pvalue7, decip)," \n")
    cat("\n")
    cat("   Weighted generalized score statistic for H0: NPV1 = NPV2 is ", round(T2, decip)," and the two-sided p-value is ", round(pvalue8, decip)," \n")
    cat("\n")  
    
    if (min(pvalue7, pvalue8) <= alpha / 2 & max(pvalue7, pvalue8) <= alpha)
    {
      cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: PPV1 = PPV2 and the hypothesis H0: NPV1 = NPV2 \n")
      cat("\n")
      
      if(PPV1 > PPV2)
      {
        cat("   Positive PV of Test 1 is significantly greater than positive PV of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference PPV1 - PPV2 is (", round(100 * LPPV, decip),"% ; ", round(100 * UPPV, decip),"%) \n")          
        cat("\n")
      }
      else
      {
        cat("   Positive PV of Test 2 is significantly greater than positive PV of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference PPV2 - PPV1 is (", round(100* (- UPPV), decip),"% ; ", round(100 * (- LPPV), decip),"%) \n")          
        cat("\n")     
      }
      
      if(NPV1 > NPV2)
      {
        cat("   Negative PV of Test 1 is significantly greater than negative PV of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference NPV1 - NPV2 is (", round(100 * LNPV, decip),"% ; ", round(100 * UNPV, decip),"%) \n")          
        cat("\n")
      }
      else
      {
        cat("   Negative PV of Test 2 is significantly greater than negative PV of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference NPV2 - NPV1 is (", round(100 * (- UNPV), decip),"% ; ", round(100 * (- LNPV), decip),"%) \n")          
        cat("\n")     
      }
    }
    
    if (pvalue7 <= alpha / 2 & pvalue8 > alpha)
    {
      cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we reject the hypothesis H0: PPV1 = PPV2 and we do not reject the hypothesis H0: NPV1 = NPV2 \n")
      cat("\n")
      
      if(PPV1 > PPV2)
      {
        cat("   Positive PV of Test 1 is significantly greater than positive PV of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference PPV1 - PPV2 is (", round(100 * LPPV, decip),"% ; ", round(100 * UPPV, decip),"%) \n")          
        cat("\n")
      }
      else
      {
        cat("   Positive PV of Test 2 is significantly greater than positive PV of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference PPV2 - PPV1 is (", round(100 * (- UPPV), decip),"% ; ", round(100 * (- LPPV), decip),"%) \n")          
        cat("\n")    
      }
    }
    
    if (pvalue7 > alpha & pvalue8 <= alpha / 2)
    {
      cat("   Applying the Holm method (to an alpha error of", 100 * alpha,"%), we do not reject the hypothesis H0: PPV1 = PPV2 and we reject the hypothesis H0: NPV1 = NPV2 \n")
      cat("\n")
      
      if(NPV1 > NPV2)
      {
        cat("   Negative PV of Test 1 is significantly greater than negative PV of Test 2 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference NPV1 - NPV2 is (", round(100 * LNPV, decip),"% ; ", round(100 * UNPV, decip),"%) \n")          
        cat("\n")
      }
      else
      {
        cat("   Negative PV of Test 2 is significantly greater than negative PV of Test 1 \n")
        cat("\n")                   
        cat("   ",100 * conf,"% confidence interval for the difference NPV2 - NPV1 is (", round(100 * (- UNPV), decip),"% ; ", round(100 * (- LNPV), decip),"%) \n")          
        cat("\n")                        
      }
    }              
  }              
  sink()          
  
}      

main <- function() {
  require(data.table)
  data <- fread("C:\\Users/Igor/source/repos/Master's thesis/Master's thesis/compbdt.csv", sep=",")
  N <- readLines("C:\\Users/Igor/source/repos/Master's thesis/Master's thesis/input_data/params.txt")
  for (i in 1:(as.integer(N[2]))) {
    s11 <- data[i, s11]
    s10 <- data[i, s10]
    s01 <- data[i, s01]
    s00 <- data[i, s00]
    r11 <- data[i, r11]
    r10 <- data[i, r10]
    r01 <- data[i, r01]
    r00 <- data[i, r00]
    
    compbdt(s11, s10, s01, s00, r11, r10, r01, r00)
  }
}

main()
