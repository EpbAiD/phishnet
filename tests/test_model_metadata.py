# ===============================================================
# tests/test_model_metadata.py
# Unit tests for model versioning and metadata
# ===============================================================
import pytest
import json
from datetime import datetime
from src.api.model_metadata import ModelMetadata, create_model_metadata, get_model_version_from_filename


class TestModelMetadata:
    """Test suite for ModelMetadata class."""

    def test_create_metadata(self):
        """Test creating model metadata."""
        metadata = ModelMetadata(
            version="v1.0.0",
            model_type="url",
            algorithm="catboost",
            trained_date="2025-01-01T00:00:00Z",
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            threshold=0.5,
            feature_count=39,
            training_samples=32000,
            cross_validation_folds=5,
            hyperparameters={"depth": 6, "iterations": 100}
        )

        assert metadata.version == "v1.0.0"
        assert metadata.model_type == "url"
        assert metadata.accuracy == 0.90
        assert metadata.feature_count == 39

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            version="v1.0.0",
            model_type="url",
            algorithm="catboost",
            trained_date="2025-01-01T00:00:00Z",
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            threshold=0.5,
            feature_count=39,
            training_samples=32000,
            cross_validation_folds=5,
            hyperparameters={}
        )

        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["version"] == "v1.0.0"
        assert data["accuracy"] == 0.90

    def test_metadata_to_json(self):
        """Test converting metadata to JSON."""
        metadata = ModelMetadata(
            version="v1.0.0",
            model_type="url",
            algorithm="catboost",
            trained_date="2025-01-01T00:00:00Z",
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            threshold=0.5,
            feature_count=39,
            training_samples=32000,
            cross_validation_folds=5,
            hyperparameters={}
        )

        json_str = metadata.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["version"] == "v1.0.0"

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "version": "v1.0.0",
            "model_type": "url",
            "algorithm": "catboost",
            "trained_date": "2025-01-01T00:00:00Z",
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
            "f1_score": 0.90,
            "threshold": 0.5,
            "feature_count": 39,
            "training_samples": 32000,
            "cross_validation_folds": 5,
            "hyperparameters": {}
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.version == "v1.0.0"
        assert metadata.accuracy == 0.90


class TestHelperFunctions:
    """Test helper functions for model metadata."""

    def test_create_model_metadata(self):
        """Test create_model_metadata helper."""
        metrics = {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
            "f1_score": 0.90
        }

        metadata = create_model_metadata(
            version="v1.0.0",
            model_type="url",
            algorithm="catboost",
            metrics=metrics,
            feature_count=39,
            training_samples=32000,
            hyperparameters={"depth": 6},
            threshold=0.5,
            cross_validation_folds=5
        )

        assert metadata.version == "v1.0.0"
        assert metadata.accuracy == 0.90
        assert metadata.feature_count == 39
        assert "depth" in metadata.hyperparameters

    def test_get_model_version_from_filename(self):
        """Test extracting version from filename."""
        # With version
        version = get_model_version_from_filename("url_catboost_v1.2.3.pkl")
        assert version == "v1.2.3"

        # Without version (should return default)
        version = get_model_version_from_filename("url_catboost.pkl")
        assert version == "v1.0.0"

        # Different format
        version = get_model_version_from_filename("model_v2.0.0_final.pkl")
        assert version == "v2.0.0"
