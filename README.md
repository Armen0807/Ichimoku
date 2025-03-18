README - Stratégie de Trading VolTarget avec Ichimoku
Ce projet implémente une stratégie de trading basée sur l'indicateur Ichimoku Kinko Hyo, combinée avec une gestion de risque basée sur la volatilité (VolTarget). La stratégie vise à identifier des opportunités d'entrée et de sortie sur un actif financier en utilisant les composants de l'Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span) et en ajustant les seuils de trading en fonction de la volatilité du marché.

Fonctionnalités Principales
Calcul des Composants Ichimoku :

Tenkan-sen (Conversion Line) : Moyenne des plus hauts et plus bas sur une période donnée.

Kijun-sen (Base Line) : Moyenne des plus hauts et plus bas sur une période plus longue.

Senkou Span A (Leading Span A) : Moyenne du Tenkan-sen et du Kijun-sen, décalée dans le futur.

Senkou Span B (Leading Span B) : Moyenne des plus hauts et plus bas sur une période encore plus longue, décalée dans le futur.

Chikou Span (Lagging Span) : Prix de clôture décalé dans le passé.

Gestion de la Volatilité (VolTarget) :

La stratégie ajuste les seuils de trading en fonction de la volatilité à court terme et à long terme.

Les niveaux de stop-loss et de take-profit sont dynamiquement calculés en fonction de la volatilité actuelle.

Conditions d'Entrée et de Sortie :

Entrée Longue : Lorsque le prix dépasse les Senkou Span A et B, et que le Tenkan-sen est supérieur au Kijun-sen avec un écart supérieur au seuil dynamique.

Entrée Courte : Lorsque le prix est en dessous des Senkou Span A et B, et que le Tenkan-sen est inférieur au Kijun-sen avec un écart supérieur au seuil dynamique.

Sortie : Les positions sont fermées lorsque les conditions d'entrée ne sont plus remplies ou lorsque les niveaux de stop-loss/take-profit sont atteints.

Gestion des Risques :

Les positions sont ajustées en fonction de la volatilité pour limiter l'exposition au risque.

Les seuils de trading sont recalculés régulièrement pour s'adapter aux conditions de marché changeantes.

Structure du Code
Classes Principales
VolTargetConfig :

Contient les paramètres de configuration de la stratégie, tels que les périodes pour les lignes Ichimoku, les niveaux de stop-loss/take-profit, et l'actif cible.

VolTargetHistory :

Stocke l'historique des calculs de la stratégie, y compris les niveaux Ichimoku, les signaux d'entrée/sortie, et les détails de la volatilité.

VolTargetBacktestStrategy :

Implémente la logique de la stratégie, y compris le calcul des composants Ichimoku, la gestion des positions, et les conditions d'entrée/sortie.

IchimokuTrade :

Représente une transaction spécifique à la stratégie Ichimoku, avec des informations supplémentaires telles que les niveaux d'entrée Tenkan-sen et Kijun-sen.

Utilisation
Configuration
Paramètres de la Stratégie :

conversion_line_periods : Période pour le calcul du Tenkan-sen.

base_line_periods : Période pour le calcul du Kijun-sen.

lagging_span_periods : Période pour le calcul du Chikou Span.

displacement : Décalage pour les Senkou Span A et B.

threshold : Seuils dynamiques pour les conditions d'entrée.

take_profit et stop_loss : Niveaux de profit et de perte.

asset : Actif cible pour la stratégie.

Exemple de Configuration :

python
Copy
config = VolTargetConfig(
    conversion_line_periods=9,
    base_line_periods=26,
    lagging_span_periods=52,
    displacement=26,
    threshold={"base_level": 100, "calculation_range": 10, "base_date": dt.date(2023, 1, 1)},
    take_profit=0.02,
    stop_loss=0.01,
    expected_profit=0.03,
    asset="AAPL"
)
Exécution de la Stratégie
Initialisation :

python
Copy
strategy = VolTargetBacktestStrategy(config)
Calcul des Composants Ichimoku :

Utilisez les méthodes calc_conversion_line, calc_base_line, calc_leading_span_1, calc_leading_span_2, et calc_lagging_span pour calculer les niveaux Ichimoku.

Vérification des Conditions d'Entrée/Sortie :

Utilisez check_long_entry_condition et check_short_entry_condition pour déterminer les opportunités d'entrée.

Utilisez check_long_exit_condition et check_short_exit_condition pour déterminer les opportunités de sortie.

Gestion des Positions :

Les positions sont gérées automatiquement en fonction des conditions d'entrée/sortie et des niveaux de stop-loss/take-profit.

Exemple de Workflow
python
Copy
# Initialisation
strategy = VolTargetBacktestStrategy(config)

# Simulation sur une période donnée
start_date = dt.date(2023, 1, 1)
end_date = dt.date(2023, 12, 31)
current_date = start_date

while current_date <= end_date:
    # Calcul des niveaux Ichimoku
    conversion_line = strategy.calc_conversion_line("AAPL", current_date)
    base_line = strategy.calc_base_line("AAPL", current_date)
    leading_span_1 = strategy.calc_leading_span_1("AAPL", current_date)
    leading_span_2 = strategy.calc_leading_span_2("AAPL", current_date)
    lagging_span = strategy.calc_lagging_span("AAPL", current_date)

    # Vérification des conditions d'entrée
    if strategy.check_long_entry_condition("AAPL", current_date):
        strategy.open_position = True
        logger.info(f"Long entry signal on {current_date}")

    # Vérification des conditions de sortie
    if strategy.check_long_exit_condition("AAPL", current_date):
        strategy.open_position = False
        logger.info(f"Long exit signal on {current_date}")

    # Passage au jour suivant
    current_date = strategy.calendar.busday_add(current_date, 1)
Dépendances
numpy : Pour les calculs numériques.

pydantic : Pour la validation des configurations.

loguru : Pour la journalisation des événements.

grt_lib_price_loader : Pour le chargement des données de prix.

grt_lib_orchestrator : Pour l'orchestration des stratégies de backtest.

grt_lib_order_book : Pour la gestion des ordres et des transactions.

Améliorations Possibles
Optimisation des Paramètres :

Utiliser des techniques d'optimisation pour trouver les meilleures périodes et seuils pour l'Ichimoku et la gestion de la volatilité.

Backtesting :

Implémenter un backtest complet pour évaluer la performance de la stratégie sur des données historiques.

Gestion des Risques Avancée :

Ajouter des fonctionnalités de gestion des risques, telles que la diversification du portefeuille et la gestion des corrélations entre actifs.

Visualisation :

Ajouter des graphiques pour visualiser les niveaux Ichimoku, les signaux d'entrée/sortie, et la performance de la stratégie.

Conclusion
Cette stratégie combine l'analyse technique (Ichimoku) avec une gestion de risque basée sur la volatilité (VolTarget) pour identifier des opportunités de trading sur les marchés financiers. Elle est conçue pour être flexible et adaptable à différents actifs et conditions de marché.
