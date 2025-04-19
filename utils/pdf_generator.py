from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os
from datetime import datetime

class PDFGenerator:
    def __init__(self):
        self.output_dir = 'static/reports'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_report(self, analysis_results):
        """Generate a PDF report from analysis results"""
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'deepfake_report_{timestamp}.pdf'
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        
        # Create content
        content = []
        
        # Add title
        content.append(Paragraph("Deepfake Analysis Report", title_style))
        content.append(Spacer(1, 12))
        
        # Add timestamp
        content.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        content.append(Spacer(1, 24))
        
        # Add summary section
        content.append(Paragraph("Analysis Summary", styles['Heading2']))
        content.append(Spacer(1, 12))
        
        # Add overall probability
        if 'video_analysis' in analysis_results:
            video_analysis = analysis_results['video_analysis']
            probability = video_analysis.get('deepfake_probability', 0)
            probability_text = f"Overall Deepfake Probability: {probability:.2%}"
            content.append(Paragraph(probability_text, styles['Normal']))
            
            # Add frames analyzed
            frames_text = f"Frames Analyzed: {video_analysis.get('frames_analyzed', 0)}"
            content.append(Paragraph(frames_text, styles['Normal']))
            
        content.append(Spacer(1, 24))
        
        # Add detailed analysis section
        content.append(Paragraph("Detailed Analysis", styles['Heading2']))
        content.append(Spacer(1, 12))
        
        # Add analysis points
        if 'video_analysis' in analysis_results:
            analysis_points = analysis_results['video_analysis'].get('analysis_points', [])
            if analysis_points:
                # Create table data
                table_data = [['Type', 'Confidence']]
                for point in analysis_points:
                    table_data.append([
                        point.get('type', 'Unknown'),
                        f"{point.get('confidence', 0):.2%}"
                    ])
                
                # Create table
                table = Table(table_data, colWidths=[4*inch, 2*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                content.append(table)
            else:
                content.append(Paragraph("No specific analysis points detected.", styles['Normal']))
                
        content.append(Spacer(1, 24))
        
        # Add conclusion section
        content.append(Paragraph("Conclusion", styles['Heading2']))
        content.append(Spacer(1, 12))
        
        if 'video_analysis' in analysis_results:
            probability = analysis_results['video_analysis'].get('deepfake_probability', 0)
            if probability > 0.7:
                conclusion = "The analyzed content shows strong indicators of being a deepfake."
            elif probability > 0.3:
                conclusion = "The analyzed content shows some indicators of being a deepfake."
            else:
                conclusion = "The analyzed content appears to be authentic."
                
            content.append(Paragraph(conclusion, styles['Normal']))
            
        # Build PDF
        doc.build(content)
        
        return filepath 