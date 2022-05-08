function getBotResponse() {
	// first remove the placeholder in the chat area
	$("#msg1").remove();
 	var user_input_text = $("#user_input").val();
 	$("#user_input").val("");

 	// add the user message in the chat area
 	var user_message_html = '<p class="user_message"><span>' + user_input_text + "</span></p>";

 	$("#chat_holder").append(user_message_html);
 	$.get("/get", { user_query: $.trim(user_input_text) }).done(function(bot_response_list) {
 		$.each( bot_response_list, function( index, response ) {
 			// bot response is a list of strings so each of them are placed one by one
 			if (response.trim()) {
 				var bot_response_html = '<p class="bot_message"><span>' + response.replace("\n", "</br>"); + "</span></p>";
 				$("#chat_holder").append(bot_response_html);
 				$("#chat_holder").animate({ scrollTop: $('#chat_holder').prop("scrollHeight")}, 1000);
 			}
		});
	});
}

$("#user_input").keyup(function(e) {
	// detects the enter key to send the query to bot server
	if (e.which == 13) {
		var rawText = $("#user_input").val();
		getBotResponse();
  	}
});