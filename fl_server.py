# fl_server.py
import flwr as fl

def main():

    def weighted_corr_agg(metrics):
        """Compute the weighted average of per‐client 'correlation' metrics."""
        # metrics is List[Tuple[int, Dict[str, float]]], i.e.
        #   [(num_examples_1, {"correlation": corr1}), …]
        total_samples = sum(n for n, _ in metrics)
        if total_samples == 0:
            return {"avg_correlation": 0.0}
        weighted_sum = sum(n * m["avg_correlation"] for n, m in metrics)
        return {"avg_correlation": weighted_sum / total_samples}
    
    def weighted_eval_corr(metrics):
        """
        metrics: List of (num_examples, {"correlation": value})
        Return a weighted average of the 'correlation' metric.
        """
        total = sum(n for n, _ in metrics)
        if total == 0:
            return {"avg_correlation": 0.0}
        weighted_sum = sum(n * m["avg_correlation"] for n, m in metrics)
        return {"avg_correlation": weighted_sum / total}

    # Configure FedAvg strategy:
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # ask all clients to fit each round
        # fraction_evaluate=0.0,      # no centralized evaluation
        min_fit_clients=2,          # wait for 3 clients to join
        min_evaluate_clients=0,     # since fraction_evaluate=0.0
        min_available_clients=2,  
        fit_metrics_aggregation_fn=weighted_corr_agg,  # before starting
        evaluate_metrics_aggregation_fn=weighted_eval_corr, 
    )

    # Start Flower server for 5 rounds
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
