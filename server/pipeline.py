# server/pipeline.py
from server.rag import MilitaryRAG
from server.llm import ReportGenerator
from server.reports import add_metadata

class MilitaryReportPipeline:
    def __init__(self):
        self.rag = MilitaryRAG()
        self.generator = ReportGenerator()
    
    def process_alert(self, alert: dict) -> dict:
        """
        Main pipeline: Alert → Report
        """
        
        # 1. Determine report type based on alert
        report_type = self.determine_report_type(alert)
        
        # 2. Search relevant docs
        rag_results = self.rag.search(
            f"{report_type} format for {alert['type']}"
        )
        
        # 3. Generate report
        if report_type == "SPOTREP":
            report_text = self.generator.generate_spotrep(alert)
        elif report_type == "CASEVAC":
            report_text = self.generator.generate_casevac(alert)
        # ... other types
        
        # 4. Add metadata
        final_report = add_metadata(report_text, alert, rag_results)
        
        return {
            "report": report_text,
            "metadata": final_report.dict(),
            "confidence": final_report.overall_confidence,
            "needs_review": final_report.needs_human_review
        }
    
    def determine_report_type(self, alert: dict) -> str:
        """Decide which report to generate"""
        
        if "injury" in alert or "casualty" in alert:
            return "CASEVAC"
        elif alert.get("type") in ["person", "vehicle", "aircraft"]:
            return "SPOTREP"
        else:
            return "SPOTREP"  # Default