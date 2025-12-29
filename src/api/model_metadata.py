# ===============================================================
# src/api/model_metadata.py
# Model Versioning and Metadata Management
# ===============================================================
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """
    Model metadata for versioning and tracking.
    """
    version: str
    model_type: str  # "url", "whois", "dns"
    algorithm: str  # "catboost", "xgboost", etc.
    trained_date: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    threshold: float
    feature_count: int
    training_samples: int
    cross_validation_folds: int
    hyperparameters: Dict[str, Any]
    feature_importance_top_10: Optional[Dict[str, float]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create ModelMetadata from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Create ModelMetadata from JSON string."""
        return cls.from_dict(json.loads(json_str))


class VersionedModel:
    """
    Wrapper for model with metadata and versioning.
    """

    def __init__(self, model: Any, metadata: ModelMetadata, feature_columns: list):
        self.model = model
        self.metadata = metadata
        self.feature_columns = feature_columns
        self.loaded_at = datetime.utcnow().isoformat() + 'Z'

    def predict(self, X):
        """Proxy to model.predict()."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Proxy to model.predict_proba()."""
        return self.model.predict_proba(X)

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "metadata": self.metadata.to_dict(),
            "feature_count": len(self.feature_columns),
            "loaded_at": self.loaded_at,
            "model_class": type(self.model).__name__
        }

    def save(self, path: Path):
        """
        Save model with metadata.

        Args:
            path: Path to save model (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = path.with_suffix('.pkl')
        joblib.dump(self.model, model_file)
        logger.info(f"Model saved to {model_file}")

        # Save metadata
        metadata_file = path.with_suffix('.meta.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                "metadata": self.metadata.to_dict(),
                "feature_columns": self.feature_columns,
                "saved_at": datetime.utcnow().isoformat() + 'Z'
            }, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")

        # Save feature columns
        features_file = path.with_suffix('.features.txt')
        with open(features_file, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        logger.info(f"Feature columns saved to {features_file}")

    @classmethod
    def load(cls, path: Path) -> 'VersionedModel':
        """
        Load model with metadata.

        Args:
            path: Path to model file (with or without extension)

        Returns:
            VersionedModel instance
        """
        path = Path(path)

        # Load model
        model_file = path if path.suffix == '.pkl' else path.with_suffix('.pkl')
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model = joblib.load(model_file)
        logger.info(f"Model loaded from {model_file}")

        # Load metadata
        metadata_file = path.with_suffix('.meta.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                meta_data = json.load(f)
                metadata = ModelMetadata.from_dict(meta_data['metadata'])
                feature_columns = meta_data.get('feature_columns', [])
            logger.info(f"Metadata loaded from {metadata_file}")
        else:
            # Fallback: create basic metadata
            logger.warning(f"Metadata file not found: {metadata_file}. Using defaults.")
            metadata = ModelMetadata(
                version="unknown",
                model_type="unknown",
                algorithm=type(model).__name__,
                trained_date="unknown",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                threshold=0.5,
                feature_count=0,
                training_samples=0,
                cross_validation_folds=0,
                hyperparameters={}
            )

            # Try to load feature columns from .txt file
            features_file = path.with_suffix('.features.txt')
            if features_file.exists():
                with open(features_file, 'r') as f:
                    feature_columns = [line.strip() for line in f if line.strip()]
            else:
                feature_columns = []

        return cls(model, metadata, feature_columns)


# ===============================================================
# Helper Functions
# ===============================================================
def create_model_metadata(
    version: str,
    model_type: str,
    algorithm: str,
    metrics: Dict[str, float],
    feature_count: int,
    training_samples: int,
    hyperparameters: Dict[str, Any],
    threshold: float = 0.5,
    cross_validation_folds: int = 5,
    notes: Optional[str] = None
) -> ModelMetadata:
    """
    Create model metadata from training results.

    Args:
        version: Model version (e.g., "v1.0.0")
        model_type: Type of model ("url", "whois", "dns")
        algorithm: Algorithm name ("catboost", "xgboost", etc.)
        metrics: Dict with keys: accuracy, precision, recall, f1_score
        feature_count: Number of features
        training_samples: Number of training samples
        hyperparameters: Model hyperparameters
        threshold: Classification threshold
        cross_validation_folds: Number of CV folds used
        notes: Optional notes

    Returns:
        ModelMetadata instance
    """
    return ModelMetadata(
        version=version,
        model_type=model_type,
        algorithm=algorithm,
        trained_date=datetime.utcnow().isoformat() + 'Z',
        accuracy=metrics.get('accuracy', 0.0),
        precision=metrics.get('precision', 0.0),
        recall=metrics.get('recall', 0.0),
        f1_score=metrics.get('f1_score', 0.0),
        threshold=threshold,
        feature_count=feature_count,
        training_samples=training_samples,
        cross_validation_folds=cross_validation_folds,
        hyperparameters=hyperparameters,
        notes=notes
    )


def get_model_version_from_filename(filename: str) -> str:
    """
    Extract version from model filename.

    Example:
        "url_catboost_v1.2.3.pkl" -> "v1.2.3"
        "url_catboost.pkl" -> "v1.0.0" (default)
    """
    import re
    match = re.search(r'v\d+\.\d+\.\d+', filename)
    return match.group(0) if match else "v1.0.0"
