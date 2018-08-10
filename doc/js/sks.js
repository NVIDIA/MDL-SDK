 function frame_resize(){var height=$(window).innerHeight();var header_height=$('#blackheader').innerHeight();
 var footer_height=$('#blackfooter').innerHeight();var content_height=height - header_height - footer_height;
 $('#content').innerHeight(content_height);}function set_imagecode_width(){$(".tdimage").each(function(index){
 var code=$(this).parent().parent().find(".tdcode");var available_width=$("#page_content").outerWidth();
 var code_width=$(code).outerWidth();var aspect=parseFloat($(this).children(".rimg-box").attr("data-aspect"));
 var new_image_width=available_width - code_width;new_image_width=Math.min(new_image_width,
 parseFloat($(this).find("img").parent().attr("data-width")));$(this).find("img").innerWidth(new_image_width);
 $(this).find("img").innerHeight(new_image_width/aspect);});}function set_imagetext_width(){
 $(".imagetext").each(function(index){$(this).css("background-color","lightblue");var image=$(this).find(".tdimage");
 var text=$(this).find(".tdtext");var available_width=$("#page_content").innerWidth();var new_image_width=available_width - $(text).outerWidth();
 var aspect=parseFloat($(this).children(".rimg-box").attr("data-aspect"));$(image).innerWidth(new_image_width);
 $(image).innerHeight(new_image_width/aspect);});}function adjust_slanted_header_table(){var angle=Math.PI/3;
 $("table").each(function(){var slant_select="td[data-slant='1'] div span";if($(slant_select).length==0)
 return;table=this;var max_slant_height=0;$(table).find(slant_select).each(function(){max_slant_height=Math.max
(max_slant_height,$(this).outerWidth()* Math.sin(angle)-  $(this).closest("td").outerHeight());});
 var translate="translate3d(0px,"+max_slant_height+"px,0px)";var container=$(table).closest(".tablecontainer");
 if($(container).length==1){$(container).css("transform",translate);$(container).css("margin-bottom",max_slant_height+"px");}else{
 $(table).css("transform",translate);$(table).css("margin-bottom",(max_slant_height+10)+"px");}});}
 function adjust_footnotes(){$(".footnote .footnotetext").each(function(index){var left=$(this).parent().position().left;
 var c=$("#page_content");var half_width=$(c).width()/2.0;var mid=$(c).position().left+half_width;
 var max_width=300;if(left>mid){$(this).css("right","0em");$(this).css("left","auto");}else{
 $(this).css("left","0em");$(this).css("right","auto");}$(this).width(Math.min(max_width,half_width));});}
 var max_box_width_without_scaling=400;var max_box_width_proportional_to_code=2.2;function describe(element,comment){
}function annotate_debug(){var annotate=$(".annotatebox");annotate.css("background-color","lightblue");
 annotate.find(".codecomment").css("background-color","rgb(100%,100%,90%)");annotate.find(".uncommented")
 .css("background-color","rgb(95%,100%,90%)");}function right_pos(element){return $(element).offset().left+$(element).outerWidth();}
 function code_width(box){var result=0;$(box).find(".codeblock").each(function(index){result=Math.max(result,right_pos($(this)));});
 $(box).find(".uncommented").each(function(index){result=Math.max(result,$(this).outerWidth());});
 return result;}function set_annotate_max_width(){$(".annotatebox").each(function(index){var max_x=0;
 var box=$(this);$(box).width(2000);$(box).find(".annotatedcode").each(function(index){var code=$(this);
 var width=0;$(code).children().each(function(){width+=$(this).outerWidth();});max_x=Math.max(max_x,width);});
 $(box).find(".uncommented").each(function(index){max_x=Math.max(max_x,$(this).outerWidth());});
 max_x+=20;var scale=parseFloat($(this).attr("data-scale"));max_x *=scale;var current_max_width=$(box).attr("data-maxwidth");
 if(typeof current_max_width==="undefined"){$(box).attr("data-maxwidth",max_x);}else if(current_max_width>max_x){
 max_x=current_max_width;$(box).attr("data-maxwidth",max_x);}$(box).innerWidth(max_x);});}function find_parent_inner_width(element){
 var result=element;var in_table=false;var width;var box_offset=parseFloat($(".annotatebox").css("margin-left"))+
 parseFloat($(".annotatebox").css("margin-right"))+parseFloat($(".annotatepad").css("padding-left"))+
 parseFloat($(".annotatepad").css("padding-right"));while(true){result=$(result).parent();
 var tag=$(result).prop("tagName");if(tag=="TD" || tag=="DD"){in_table=true;break;}
 if($(result).attr("id")=="page_content"){break;}if(tag=="BODY"){break;}}if(in_table){
 var other_cell=$(result).closest("tr").children("td:first");var cell_width=$(other_cell).outerWidth()+
 parseFloat($(other_cell).css("padding-left"))+parseFloat($(other_cell).css("padding-right"))+
 parseFloat($(other_cell).css("margin-left"))+parseFloat($(other_cell).css("margin-right"));width=$("#page_content").innerWidth()- cell_width - box_offset;}else{
 width=result.width()- parseFloat(result.css("margin-left"))- parseFloat(result.css("margin-right"))-
 parseFloat(result.css("padding-left"))- parseFloat(result.css("padding-right"))- box_offset;}
 return width;}function set_code_comment_width(){$(".codecomment").each(function(index){var box=$(this).closest(".annotatebox");
 if(box.attr("data-fixed")==="true"){$(this).css("display","inline").css("float","none");}else{
 var scale=parseFloat($(box).attr("data-scale"));var parent_inner_width=find_parent_inner_width($(box));
 var new_box_width=Math.min(parseFloat($(box).attr("data-maxwidth")),parent_inner_width);$(box).width(new_box_width);
 var available_width=$(box).innerWidth();var indent_width=$(this).siblings(".codeindent").outerWidth();
 var code_width=$(this).siblings(".codeblock").outerWidth()+1;var box_pad=2 * parseFloat(
 $(this).closest(".annotatepad").css("padding-left"));var comment_width=available_width - indent_width - code_width - box_pad;
 comment_width+=1;if(comment_width<8){$(this).hide();$(this).siblings(".codeindent").hide();}else{
 $(this).siblings(".codeindent").show();$(this).show();$(this).innerWidth(comment_width);}}});}
 $(window).on("load",function(){$('#content').css('background-color','white');$(window).resize(function(){
 frame_resize();});frame_resize();});


