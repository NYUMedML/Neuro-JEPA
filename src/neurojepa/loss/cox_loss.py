# Cox Partial Likelihood Loss (DeepSurv Loss)
import torch

class CoxPHLoss(torch.nn.Module):
    """
    Negative partial log-likelihood for Cox Proportional Hazards model.
    
    The loss is computed as:
        L = -1/N * sum_{i: event_i=1} [ h_i - log(sum_{j in R_i} exp(h_j)) ] + l2_lambda * mean(h^2)
    
    where:
        - h_i is the predicted hazard (risk score) for sample i
        - R_i is the risk set at time t_i (all samples with survival time >= t_i)
        - event_i indicates if sample i experienced the event (1) or was censored (0)
        - l2_lambda controls the strength of L2 regularization on risk scores
    """
    
    def __init__(self, reduction: str = 'mean', l2_lambda: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.l2_lambda = l2_lambda
    
    def forward(self, risk_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Args:
            risk_pred: (B,) Predicted risk scores (log-hazard)
            time: (B,) Survival/censoring times
            event: (B,) Event indicators (1 = event, 0 = censored)
        
        Returns:
            loss: Scalar tensor
        """
        # Sort by descending time for efficient risk set computation
        sorted_indices = torch.argsort(time, descending=True)
        sorted_risk = risk_pred[sorted_indices]
        sorted_event = event[sorted_indices]
        
        # Compute log-sum-exp of risk scores for the risk set
        # Using cumulative sum for efficiency
        max_risk = sorted_risk.max()
        exp_risk = torch.exp(sorted_risk - max_risk)
        cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
        log_risk_set = torch.log(cumsum_exp_risk + 1e-7) + max_risk
        
        # Partial log-likelihood (only for uncensored samples)
        uncensored_likelihood = sorted_risk - log_risk_set
        censored_mask = sorted_event.bool()
        
        if censored_mask.sum() == 0:
            # No events in batch - return zero loss
            return risk_pred.sum() * 0.0
        
        neg_log_likelihood = -uncensored_likelihood[censored_mask]
        
        if self.reduction == 'mean':
            cox_loss = neg_log_likelihood.mean()
        elif self.reduction == 'sum':
            cox_loss = neg_log_likelihood.sum()
        else:
            return neg_log_likelihood
        
        # L2 regularization on risk scores to prevent extreme predictions
        if self.l2_lambda > 0:
            l2_penalty = self.l2_lambda * risk_pred.pow(2).mean()
            cox_loss = cox_loss + l2_penalty
        
        return cox_loss
        
        
# Concordance Index (C-Index) Metric

def concordance_index(risk_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    """
    Compute the concordance index (C-index) for survival predictions.
    
    C-index measures the proportion of concordant pairs among all comparable pairs.
    A pair (i, j) is comparable if the sample with shorter survival time had an event.
    A pair is concordant if the sample with higher risk has shorter survival time.
    
    Args:
        risk_pred: (N,) Predicted risk scores (higher = higher risk)
        time: (N,) Survival/censoring times
        event: (N,) Event indicators (1 = event, 0 = censored)
    
    Returns:
        c_index: Float in [0, 1], where 0.5 is random, 1.0 is perfect
    """
    # Convert to float32 before numpy (BFloat16 not supported by numpy)
    risk_pred = risk_pred.detach().cpu().float().numpy()
    time = time.detach().cpu().float().numpy()
    event = event.detach().cpu().float().numpy()
    
    n = len(time)
    concordant = 0
    discordant = 0
    tied_risk = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only consider comparable pairs (one must have event)
            if time[i] == time[j]:
                continue
            
            # Determine which has shorter time
            if time[i] < time[j]:
                shorter_idx, longer_idx = i, j
            else:
                shorter_idx, longer_idx = j, i
            
            # Only comparable if shorter time had event
            if event[shorter_idx] == 0:
                continue
            
            # Check concordance
            if risk_pred[shorter_idx] > risk_pred[longer_idx]:
                concordant += 1
            elif risk_pred[shorter_idx] < risk_pred[longer_idx]:
                discordant += 1
            else:
                tied_risk += 0.5
    
    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5  # No comparable pairs
    
    return (concordant + tied_risk) / total


def fast_concordance_index(risk_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    """
    Faster C-index computation using vectorized operations.
    Suitable for larger datasets.
    """
    # Convert to float32 before numpy (BFloat16 not supported by numpy)
    risk_pred = risk_pred.detach().cpu().float().numpy()
    time = time.detach().cpu().float().numpy()
    event = event.detach().cpu().float().numpy()
    
    n = len(time)
    
    # Create comparison matrices
    time_diff = time[:, None] - time[None, :]  # time_i - time_j
    risk_diff = risk_pred[:, None] - risk_pred[None, :]  # risk_i - risk_j
    
    # Valid pairs: i has event AND time_i < time_j (j is at risk when i has event)
    valid_pairs = (event[:, None] == 1) & (time_diff < 0)
    
    # Concordant: risk_i > risk_j (higher risk, shorter time)
    concordant = (valid_pairs & (risk_diff > 0)).sum()
    
    # Discordant: risk_i < risk_j
    discordant = (valid_pairs & (risk_diff < 0)).sum()
    
    # Tied risk
    tied = (valid_pairs & (risk_diff == 0)).sum() * 0.5
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return float((concordant + tied) / total)