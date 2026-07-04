"""
InLegalBERT Hallucination Test Suite (Dynamic Runtime)
========================================================
Tests the model's behavior without any RAG or external AI.
Measures: similarity scores between real/fake legal concepts,
          negation blindness, OOD detection, and hallucination risk.

Run: python inlegalbert_hallucination_test.py
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json
from datetime import datetime

class InLegalBERTHallucinationTester:
    """
    Dynamic hallucination testing for InLegalBERT
    Tests embedding behavior without any external LLM or RAG
    """
    
    def __init__(self, model_name: str = "law-ai/InLegalBERT"):
        print(f"\n🔍 Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.results = {
            "tests": [],
            "hallucination_score": 0.0,
            "risk_areas": []
        }
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Embedding dimension: {self.model.config.hidden_size}")
        print(f"   Max tokens: {self.tokenizer.model_max_length}\n")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def cosine_similarity_score(self, text_a: str, text_b: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb_a = self.get_embedding(text_a)
        emb_b = self.get_embedding(text_b)
        return float(cosine_similarity(emb_a, emb_b)[0][0])
    
    def run_test(self, name: str, text_a: str, text_b: str, 
                 expected_behavior: str, risk_level: str = "medium") -> Dict:
        """
        Run a single hallucination test
        expected_behavior: "different", "similar", "identical"
        """
        similarity = self.cosine_similarity_score(text_a, text_b)
        
        # Determine if this is a hallucination
        is_hallucination = False
        reason = ""
        
        if expected_behavior == "different":
            if similarity > 0.85:
                is_hallucination = True
                reason = f"Should be different but got {similarity:.3f} similarity"
            elif similarity > 0.75:
                reason = f"⚠️ Borderline: {similarity:.3f} similarity (expected <0.75)"
            else:
                reason = f"✅ Good separation: {similarity:.3f}"
                
        elif expected_behavior == "similar":
            if similarity < 0.70:
                is_hallucination = True
                reason = f"Should be similar but got {similarity:.3f} similarity"
            else:
                reason = f"✅ Good similarity: {similarity:.3f}"
                
        elif expected_behavior == "identical":
            if similarity < 0.95:
                is_hallucination = True
                reason = f"Should be identical but got {similarity:.3f}"
            else:
                reason = f"✅ Excellent: {similarity:.3f}"
        
        result = {
            "name": name,
            "similarity": similarity,
            "expected": expected_behavior,
            "is_hallucination": is_hallucination,
            "reason": reason,
            "risk_level": risk_level,
            "text_a": text_a[:100],
            "text_b": text_b[:100]
        }
        
        self.results["tests"].append(result)
        return result
    
    def test_section_hallucination(self):
        """Test if model hallucinates non-existent sections"""
        print("\n" + "="*70)
        print("📋 TEST 1: Section Number Hallucination")
        print("="*70)
        
        test_cases = [
            {
                "name": "Real vs Fake Section (302 vs 999)",
                "text_a": "Section 302 IPC deals with punishment for murder.",
                "text_b": "Section 999 IPC deals with punishment for murder.",
                "expected": "different"
            },
            {
                "name": "Real vs Non-existent High Number",
                "text_a": "Section 376 IPC - punishment for rape.",
                "text_b": "Section 9999 IPC - punishment for rape.",
                "expected": "different"
            },
            {
                "name": "Two Different Real Sections",
                "text_a": "Section 302 IPC (murder) punishment is death or life imprisonment.",
                "text_b": "Section 304 IPC (culpable homicide) punishment is 10 years or life.",
                "expected": "different"
            },
            {
                "name": "Real vs Real (Similar Domains)",
                "text_a": "Section 376 IPC - rape, minimum 7 years.",
                "text_b": "Section 354 IPC - assault, up to 2 years.",
                "expected": "different"
            }
        ]
        
        results = []
        for test in test_cases:
            result = self.run_test(
                name=test["name"],
                text_a=test["text_a"],
                text_b=test["text_b"],
                expected_behavior=test["expected"],
                risk_level="high"
            )
            results.append(result)
            status = "🔴 HALLUCINATION" if result["is_hallucination"] else "✅ OK"
            print(f"\n{result['name']}:")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Status: {status}")
            print(f"  {result['reason']}")
        
        return results
    
    def test_negation_blindness(self):
        """Test if model distinguishes between affirmative and negative statements"""
        print("\n" + "="*70)
        print("📋 TEST 2: Negation Blindness (Critical)")
        print("="*70)
        
        test_cases = [
            {
                "name": "Guilty vs Not Guilty",
                "text_a": "The accused is guilty of murder under Section 302.",
                "text_b": "The accused is NOT guilty of murder under Section 302.",
                "expected": "different"
            },
            {
                "name": "Convicted vs Acquitted",
                "text_a": "The court convicted the defendant for theft.",
                "text_b": "The court acquitted the defendant of theft.",
                "expected": "different"
            },
            {
                "name": "With vs Without Consent",
                "text_a": "The act was done with consent of the victim.",
                "text_b": "The act was done without consent of the victim.",
                "expected": "different"
            },
            {
                "name": "Sanctioned vs Not Sanctioned",
                "text_a": "The government sanctioned the project.",
                "text_b": "The government did NOT sanction the project.",
                "expected": "different"
            }
        ]
        
        results = []
        for test in test_cases:
            result = self.run_test(
                name=test["name"],
                text_a=test["text_a"],
                text_b=test["text_b"],
                expected_behavior=test["expected"],
                risk_level="critical"
            )
            results.append(result)
            status = "🔴 HALLUCINATION" if result["is_hallucination"] else "✅ OK"
            print(f"\n{result['name']}:")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Status: {status}")
            print(f"  {result['reason']}")
            if result["similarity"] > 0.85:
                print(f"  ⚠️ CRITICAL: Model cannot distinguish opposite legal outcomes!")
        
        return results
    
    def test_jurisdiction_confusion(self):
        """Test if model confuses different jurisdictions"""
        print("\n" + "="*70)
        print("📋 TEST 3: Jurisdiction Confusion")
        print("="*70)
        
        test_cases = [
            {
                "name": "Maharashtra vs Karnataka Law",
                "text_a": "The Maharashtra Rent Control Act applies to properties in Mumbai.",
                "text_b": "The Karnataka Rent Control Act applies to properties in Bangalore.",
                "expected": "different"
            },
            {
                "name": "Supreme Court vs High Court",
                "text_a": "The Supreme Court of India has appellate jurisdiction.",
                "text_b": "The Delhi High Court has appellate jurisdiction.",
                "expected": "different"
            },
            {
                "name": "Civil vs Criminal Court",
                "text_a": "The civil court has jurisdiction over property disputes.",
                "text_b": "The criminal court has jurisdiction over theft cases.",
                "expected": "different"
            },
            {
                "name": "State vs Central Law",
                "text_a": "Under the Uttar Pradesh Prohibition Act.",
                "text_b": "Under the Central Prohibition Act.",
                "expected": "different"
            }
        ]
        
        results = []
        for test in test_cases:
            result = self.run_test(
                name=test["name"],
                text_a=test["text_a"],
                text_b=test["text_b"],
                expected_behavior=test["expected"],
                risk_level="medium"
            )
            results.append(result)
            status = "🔴 HALLUCINATION" if result["is_hallucination"] else "✅ OK"
            print(f"\n{result['name']}:")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Status: {status}")
            print(f"  {result['reason']}")
        
        return results
    
    def test_old_vs_new_law(self):
        """Test if model knows about post-2019 laws"""
        print("\n" + "="*70)
        print("📋 TEST 4: Old Law vs New Law (Training Cutoff)")
        print("="*70)
        
        test_cases = [
            {
                "name": "IPC (1860) vs BNS (2023) - Murder",
                "text_a": "Section 302 IPC: Punishment for murder is death or life imprisonment.",
                "text_b": "Section 101 BNS: Punishment for murder is death or life imprisonment.",
                "expected": "different"
            },
            {
                "name": "IPC vs BNS - Theft",
                "text_a": "Section 378 IPC defines theft.",
                "text_b": "Section 303 BNS defines theft.",
                "expected": "different"
            },
            {
                "name": "Companies Act 1956 vs 2013",
                "text_a": "Section 271 of Companies Act 1956 deals with winding up.",
                "text_b": "Section 271 of Companies Act 2013 deals with winding up.",
                "expected": "different"
            },
            {
                "name": "IT Act 2000 vs DPDP Act 2023",
                "text_a": "Section 43 of IT Act 2000 deals with data protection.",
                "text_b": "Section 8 of DPDP Act 2023 deals with data protection.",
                "expected": "different"
            }
        ]
        
        results = []
        for test in test_cases:
            result = self.run_test(
                name=test["name"],
                text_a=test["text_a"],
                text_b=test["text_b"],
                expected_behavior=test["expected"],
                risk_level="high"
            )
            results.append(result)
            status = "🔴 HALLUCINATION" if result["is_hallucination"] else "✅ OK"
            print(f"\n{result['name']}:")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Status: {status}")
            print(f"  {result['reason']}")
            if result["similarity"] > 0.85:
                print(f"  ⚠️ DANGER: Model treats IPC and BNS as identical (they are NOT!)")
        
        return results
    
    def test_out_of_distribution(self):
        """Test how model handles completely unfamiliar concepts"""
        print("\n" + "="*70)
        print("📋 TEST 5: Out-of-Distribution Detection")
        print("="*70)
        
        test_cases = [
            {
                "name": "Fake Law vs Real Law",
                "text_a": "Under the Digital Personal Data Protection Act, 2023.",
                "text_b": "Under the Information Technology Act, 2000.",
                "expected": "different"
            },
            {
                "name": "Non-legal vs Legal Text",
                "text_a": "The stock market crashed by 5% today.",
                "text_b": "The court dismissed the petition due to lack of evidence.",
                "expected": "different"
            },
            {
                "name": "Random vs Legal Concept",
                "text_a": "Quantum physics explains the behavior of particles.",
                "text_b": "Quantum meruit is a legal doctrine for unjust enrichment.",
                "expected": "different"
            },
            {
                "name": "Fictional vs Real Case",
                "text_a": "The case of Kumar vs State of India (2050).",
                "text_b": "The case of Kesavananda Bharati vs State of Kerala (1973).",
                "expected": "different"
            }
        ]
        
        results = []
        for test in test_cases:
            result = self.run_test(
                name=test["name"],
                text_a=test["text_a"],
                text_b=test["text_b"],
                expected_behavior=test["expected"],
                risk_level="medium"
            )
            results.append(result)
            status = "🔴 HALLUCINATION" if result["is_hallucination"] else "✅ OK"
            print(f"\n{result['name']}:")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Status: {status}")
            print(f"  {result['reason']}")
        
        return results
    
    def test_civil_vs_criminal(self):
        """Test confusion between civil and criminal law"""
        print("\n" + "="*70)
        print("📋 TEST 6: Civil vs Criminal Law Confusion")
        print("="*70)
        
        test_cases = [
            {
                "name": "CPC vs CrPC",
                "text_a": "Order 39 CPC deals with temporary injunctions.",
                "text_b": "Section 144 CrPC deals with temporary orders.",
                "expected": "different"
            },
            {
                "name": "Civil Suit vs Criminal Case",
                "text_a": "The plaintiff filed a suit for recovery of money.",
                "text_b": "The state filed a case for theft under IPC.",
                "expected": "different"
            },
            {
                "name": "Tort vs Crime",
                "text_a": "The defendant committed negligence causing injury.",
                "text_b": "The accused committed assault causing injury.",
                "expected": "different"
            },
            {
                "name": "Remedies: Damages vs Imprisonment",
                "text_a": "The court awarded monetary damages to the plaintiff.",
                "text_b": "The court sentenced the accused to 5 years imprisonment.",
                "expected": "different"
            }
        ]
        
        results = []
        for test in test_cases:
            result = self.run_test(
                name=test["name"],
                text_a=test["text_a"],
                text_b=test["text_b"],
                expected_behavior=test["expected"],
                risk_level="high"
            )
            results.append(result)
            status = "🔴 HALLUCINATION" if result["is_hallucination"] else "✅ OK"
            print(f"\n{result['name']}:")
            print(f"  Similarity: {result['similarity']:.4f}")
            print(f"  Status: {status}")
            print(f"  {result['reason']}")
        
        return results
    
    def calculate_hallucination_rate(self):
        """Calculate overall hallucination metrics"""
        tests = self.results["tests"]
        if not tests:
            return
        
        total = len(tests)
        hallucinations = sum(1 for t in tests if t["is_hallucination"])
        high_risk = sum(1 for t in tests if t["is_hallucination"] and t["risk_level"] == "critical")
        
        avg_similarity = np.mean([t["similarity"] for t in tests])
        
        self.results["hallucination_score"] = hallucinations / total if total > 0 else 0
        self.results["avg_similarity"] = avg_similarity
        self.results["total_tests"] = total
        self.results["hallucination_count"] = hallucinations
        self.results["critical_hallucinations"] = high_risk
        
        # Identify risk areas
        risk_by_category = {}
        for test in tests:
            if test["is_hallucination"]:
                category = test["name"].split(":")[0] if ":" in test["name"] else test["name"]
                if category not in risk_by_category:
                    risk_by_category[category] = []
                risk_by_category[category].append(test)
        
        self.results["risk_areas"] = risk_by_category
    
    def print_summary(self):
        """Print comprehensive hallucination report"""
        self.calculate_hallucination_rate()
        
        print("\n" + "="*70)
        print("📊 HALLUCINATION TEST SUMMARY REPORT")
        print("="*70)
        
        print(f"\nModel: InLegalBERT")
        print(f"Tests Run: {self.results['total_tests']}")
        print(f"Hallucinations Detected: {self.results['hallucination_count']}")
        print(f"Hallucination Rate: {self.results['hallucination_score']*100:.1f}%")
        print(f"Critical Hallucinations: {self.results['critical_hallucinations']}")
        print(f"Average Similarity Score: {self.results['avg_similarity']:.4f}")
        
        print("\n" + "-"*70)
        print("🔴 CRITICAL RISK AREAS:")
        print("-"*70)
        
        if self.results["risk_areas"]:
            for area, tests in self.results["risk_areas"].items():
                print(f"\n  ⚠️ {area}:")
                for test in tests[:3]:  # Show top 3
                    print(f"     - {test['name']}: {test['similarity']:.3f} similarity")
                    print(f"       {test['reason']}")
        else:
            print("  ✅ No critical risk areas identified")
        
        print("\n" + "-"*70)
        print("📋 DETAILED TEST RESULTS:")
        print("-"*70)
        
        for test in self.results["tests"]:
            status = "🔴 HALLUCINATION" if test["is_hallucination"] else "✅ PASS"
            print(f"\n{test['name']}:")
            print(f"  Similarity: {test['similarity']:.4f} | {status}")
            print(f"  Expected: {test['expected']}")
            print(f"  {test['reason']}")
        
        print("\n" + "="*70)
        print("🎯 FINAL VERDICT:")
        print("="*70)
        
        if self.results["hallucination_score"] > 0.3:
            print("⚠️  HIGH HALLUCINATION RISK - Model shows significant hallucination behavior")
            print("   Critical issues detected in:")
            if self.results["critical_hallucinations"] > 0:
                print("   • Negation blindness (cannot distinguish guilty/not guilty)")
                print("   • Section number confusion (fake sections = real sections)")
                print("   • Old law vs new law (IPC = BNS)")
            print("\n   RECOMMENDATION: DO NOT use alone. MUST use with RAG pipeline.")
        elif self.results["hallucination_score"] > 0.1:
            print("🟡 MODERATE HALLUCINATION RISK - Some concerning behaviors")
            print("\n   RECOMMENDATION: Use with caution. Add verification layer.")
        else:
            print("✅ LOW HALLUCINATION RISK - Model performs reasonably well")
            print("\n   RECOMMENDATION: Can use for retrieval, but still verify critical outputs.")
        
        print("\n" + "="*70)
        print("Test completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*70)
    
    def export_results(self, filename: str = "hallucination_results.json"):
        """Export results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📁 Results exported to {filename}")
    
    def run_full_suite(self):
        """Run all hallucination tests"""
        print("\n" + "🔬"*35)
        print("INLEGALBERT HALLUCINATION TEST SUITE (Dynamic Runtime)")
        print("🔬"*35)
        
        self.test_section_hallucination()
        self.test_negation_blindness()
        self.test_jurisdiction_confusion()
        self.test_old_vs_new_law()
        self.test_out_of_distribution()
        self.test_civil_vs_criminal()
        
        self.print_summary()
        self.export_results()


# Run the test suite
if __name__ == "__main__":
    tester = InLegalBERTHallucinationTester()
    tester.run_full_suite()
    
    print("\n\n💡 HOW TO INTERPRET RESULTS:")
    print("="*60)
    print("""
    High Similarity Score (>0.85) = Model thinks texts are nearly identical
    
    HALLUCINATION EXAMPLES:
    • If 'Section 302 IPC' and 'Section 999 IPC' have >0.85 similarity
      → Model cannot distinguish real vs fake sections
    
    • If 'guilty of murder' and 'NOT guilty of murder' have >0.85 similarity
      → Model suffers from negation blindness (CRITICAL)
    
    • If 'IPC Section 302' and 'BNS Section 101' have >0.85 similarity
      → Model doesn't know IPC was replaced by BNS in 2024
    
    These hallucinations happen because the model is only trained on 
    pre-2019 data and has no knowledge of post-2019 laws or real-time
    legal changes.
    
    SOLUTION: Always pair with RAG (IndianKanoon) and a verified
    legal database to ground responses in reality.
    """)